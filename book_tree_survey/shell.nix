with import <nixpkgs> {};

let
  py = python3;
in
mkShell {
  buildInputs = [

    entr

    (py.withPackages (ps: with ps; [

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
  '';
}
