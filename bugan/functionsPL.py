import os
import io

import trimesh
import numpy as np
import torch
import wandb
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#####
#   DataModule
#####
class DataModule(pl.LightningDataModule):

    def __init__(self, config, run):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None

    def prepare_data(self):
        # download
        run = self.run
        dataset_name = self.config.dataset
        self.dataset_artifact = run.use_artifact(dataset_name, type='dataset')


    def setup(self, stage=None):
        config = self.config
        dir_dict = self.dataset_artifact.metadata['dir_dict']
        artifact_dir = self.dataset_artifact.download()

        #process
        dataset = []
        for data_cat in dir_dict:
            filename_list = dir_dict[data_cat]
            for filename in filename_list:
                filename = artifact_dir + "/" + data_cat + "/" + filename
                m = trimesh.load(filename, force='mesh')
                #augment data
                if config.data_augmentation:
                    array = data_augmentation(m, num_augment_data = config.num_augment_data, array_length = config.array_size)
                else:
                    array = mesh2arrayCentered(m, array_length = config.array_size)[np.newaxis, :, :, :]
                dataset.append(array)
                
        #now all the returned array contains multiple samples
        dataset = np.concatenate(dataset)
        self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)

    def train_dataloader(self):
        config = self.config
        tensor_dataset = TensorDataset(self.dataset)
        return DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)


#####
#   functions
#####

def load_dataset(dataset_name, run, config):
    #download dataset
    dataset_artifact = run.use_artifact(dataset_name, type='dataset')
    dir_dict = dataset_artifact.metadata['dir_dict']
    artifact_dir = dataset_artifact.download()

    #process
    dataset = []
    for data_cat in dir_dict:
        filename_list = dir_dict[data_cat]
        for filename in filename_list:
            filename = artifact_dir + "/" + data_cat + "/" + filename
            m = trimesh.load(filename, force='mesh')
            #augment data
            if config.data_augmentation:
                array = data_augmentation(m, num_augment_data = config.num_augment_data, array_length = config.array_size)
            else:
                array = mesh2arrayCentered(m, array_length = config.array_size)[np.newaxis, :, :, :]
            dataset.append(array)
            
    #now all the returned array contains multiple samples
    dataset = np.concatenate(dataset)
    return dataset


def wandbLog(model, config, initial_log_dict={}, log_image=False, log_mesh=False):

    if log_image or log_mesh:
        sample_tree_array = model.generate_tree(config=config)[0]  #only 1 tree
        sample_tree_indices = netarray2indices(sample_tree_array)
        #log number of points to wandb
        print(sample_tree_indices.shape[0])
        initial_log_dict["sample_tree_numpoints"] = sample_tree_indices.shape[0]
        voxelmesh = netarray2mesh(sample_tree_array)

        if log_image:
            image = mesh2wandbImage(voxelmesh)
            initial_log_dict["sample_tree_image"] = image
            
        if log_mesh:
            voxelmeshfile = mesh2wandb3D(voxelmesh)
            initial_log_dict["sample_tree_voxelmesh"] = voxelmeshfile

    wandb.log(initial_log_dict)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

def load_model(model, model_path = 'model_dict.pth'):
    model_file = wandb.restore(model_path)
    print("restored model: "+str(model_file.name))
    model.load_state_dict(torch.load(model_file.name))
    return model

#####
#   helper function (array processing and log)
#####
def netarray2indices(array):
    coord_list = []
    if len(array.shape) == 5:
        array = array[0][0]
    x,y,z = array.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if array[i,j,k] > 0.5:        #tanh: voxel representation [-1,1], sigmoid: [0,1]
                    coord_list.append([i,j,k])
    # print(len(coord_list))
    if len(coord_list) == 0:
        return np.array([[0,0,0]])  #return at least one point to prevent wandb 3dobject error
    return np.array(coord_list)

# array should be 3d
def netarray2mesh(array):
    if len(array.shape) != 3:
        raise Exception("netarray2mesh: input array should be 3d")

    #convert to bool dtype
    array = array > 0.5
    #array all zero gives error
    if np.sum(array) == 0:
        array[0,0,0] = True
    voxelmesh = trimesh.voxel.base.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(array)).marching_cubes
    return voxelmesh

def mesh2wandbImage(voxelmesh):
    scene = voxelmesh.scene()
    try:
        png = scene.save_image(resolution=[600, 600],)
    except NoSuchDisplayException:
        print("NoSuchDisplayException. Renderer not found! Please check configuation so trimesh scene.save_image() can run successfully")
    png = io.BytesIO(png)
    image = Image.open(png)
    return wandb.Image(image)

def mesh2wandb3D(voxelmesh):
    voxelmeshfile = voxelmesh.export(file_type='obj')
    voxelmeshfile = wandb.Object3D(io.StringIO(voxelmeshfile),file_type='obj')
    return voxelmeshfile

#####
#   helper function (dataset)
#####
def mesh2arrayCentered(mesh, voxel_size = 1, array_length = 64):
    #given array length 64, voxel size 2, then output array size is [128,128,128]
    array_size = np.ceil(np.array([array_length, array_length, array_length]) / voxel_size).astype(int)
    vox_array = np.zeros(array_size, dtype=bool)    #tanh: voxel representation [-1,1], sigmoid: [0,1]
    #scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(trimesh.transformations.scale_matrix((array_length-1)/max_length))  #now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  #max voxel array length = array_length / voxel_size

    #find indices in the v.matrix to center it in vox_array
    indices = ((array_size - v.matrix.shape)/2).astype(int)
    vox_array[indices[0]:indices[0]+v.matrix.shape[0], indices[1]:indices[1]+v.matrix.shape[1], indices[2]:indices[2]+v.matrix.shape[2]] = v.matrix

    return vox_array


def data_augmentation(mesh, array_length = 64, num_augment_data = 4, scale_max_margin = 3):

    retval = np.zeros((num_augment_data, array_length, array_length, array_length))

    for i in range(num_augment_data):

        #first select rotation angle (angle in radian)
        angle = 2 * np.pi * (np.random.rand(1)[0])

        #scale is implemented based on the bounding box with box margin (larger margin, smaller scale)
        box_margin = np.random.randint(scale_max_margin + 1)

        #pick a random starting point within margin as translation
        initial_position = np.random.randint(box_margin + 1, size=3)

        result_array = modify_mesh(mesh, array_length, angle, box_margin, initial_position)
        retval[i] = result_array

    return retval


def modify_mesh(mesh, out_array_length, rot_angle, scale_box_margin, array_init_pos):
    #first copy mesh
    mesh = mesh.copy()
    #rotate mesh by rot_angle in radian
    mesh = mesh.apply_transform(trimesh.transformations.rotation_matrix(rot_angle, (0,1,0)))

    #scale is implemented based on the bounding box with box margin (larger margin, smaller scale)
        #example (assume out_array_length=64): margin = 0, bounding box shape = (64,64,64); margin = 3, bounding box shape = (61,61,61)
    scaled_size = out_array_length - scale_box_margin
    mesh_array = mesh2arrayCentered(mesh, array_length = scaled_size)

    #put them into bounding box (and translation)
    retval = np.zeros((out_array_length, out_array_length, out_array_length))
    #apply translation by selecting initial position
        #example: same mesh array of size (61,61,61) but with two position (0,1,0) and (1,0,0) is just a translation of 2 units
    x,y,z = array_init_pos
    retval[x:x+scaled_size, y:y+scaled_size, z:z+scaled_size] = mesh_array

    return retval


