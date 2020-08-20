with import <nixpkgs> {};

let
  py = python3;
in
mkShell {
   buildInputs = [

    entr
    # blender

    (py.withPackages (ps: with ps; [

      pip

      # dev deps
      ipdb
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
