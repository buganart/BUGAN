with import ../nix/nixpkgs.nix {
  overlays = [ (import ./nix/overlay.nix) ];
};

{ imageName ? "bugan-api", version ? "0.1.0" }:

let
  app = with python3.pkgs; buildPythonPackage {
    name = "app";
    propagatedBuildInputs = [
      bugan
      gunicorn
    ];
    unpackPhase = "true";
    phases = [ "installPhase" ];
    doCheck = false;
    installPhase = ''
      makeWrapper ${gunicorn}/bin/gunicorn $out/bin/app \
        --set PYTHONPATH $PYTHONPATH \
        --run 'export GUNICORN_CMD_ARGS="--bind=0.0.0.0:''${PORT:-8080} --workers=1"'
    '';
  };
  env = python3.withPackages (ps: with ps; [
    app
  ]);
in dockerTools.buildImage {
  name = imageName;
  tag = version;
  contents = [
    coreutils
    findutils
    gnugrep
    bash
  ];
  config = {
    Entrypoint = [
      "${env}/bin/app" "bugan.api:setup()"
    ];
    ExposedPorts = {
      "8080/tcp" = {};
    };
  };
}
