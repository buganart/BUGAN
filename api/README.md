### BUGAN API

Run the docker container with

    docker run -p 8080:8080 -it buganart/bugan-test-api:0.1.0

To return a sample mesh for testing:

    curl -X POST -H "Content-Type: application/json" -d '{}' 127.0.0.1:8080/generate

Or fetch and return a wavefront file from elsewhere, for example:

    curl -X POST -H "Content-Type: application/json" -d '{"url": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"}' 127.0.0.1:8080/generate

This returns a json object of the form: `{"mesh": WAVEFRONT_LINES}`,
where `WAVEFRONT_LINES` is a string containing the lines of a wavefront file:

```
# https://github.com/mikedh/trimesh
v 0.00000000 16.00000000 16.50000000
v 0.00000000 16.00000000 14.50000000
v 0.00000000 15.00000000 16.50000000
...
```
