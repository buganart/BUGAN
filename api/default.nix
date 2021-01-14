{ buildPythonPackage
, fetchFromGitHub
, certifi
, click
, flask
, pytorch
, tqdm
, requests
}:

buildPythonPackage {

  pname = "bugan";
  version = "0.1.0";
  doCheck = false;

  src = ./.;

  # TODO check if still a problem with never pytorch versions
  # ERROR: Could not find a version that satisfies the requirement dataclasses (from torch->dialog==0.1.0) (from versions: none)
  pipInstallFlags = ["--no-deps"];

  propagatedBuildInputs = [
    click
    flask

    # only used for making http request in test container
    requests

    # tqdm
    # pytorch
  ];
}
