with import ../nix/nixpkgs.nix {};

let  
  py = python3;
in
mkShell {
  buildInputs = [

    entr
    meshlab

    (py.withPackages (ps: with ps; [

      # our python deps
      click
      joblib

      # trimesh
      pillow
      networkx
      pyglet
      scipy
      shapely
      scikitimage

      # open3d python deps
      ipywidgets
      widgetsnbextension
      notebook
      numpy
      matplotlib
      trimesh

      # to install open3d
      pip  

      # dev deps
      black
      ipython
      pyls-isort
      pyls-black
      pyls-mypy
      python-language-server
    ]))
   ];

  shellHook = ''
    export PIP_PREFIX="$(pwd)/.build/pip_packages"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PIP_PREFIX/${py.sitePackages}:$PYTHONPATH"
    unset SOURCE_DATE_EPOCH

    # Runtime dependencies for open3d
    export LD_LIBRARY_PATH=${lib.makeLibraryPath [stdenv.cc.cc libGL xorg.libX11]}
  '';
}
