#!/bin/bash

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
pushd "${_SCRIPT_DIR}" # cd to script directory

# https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    : # This is not a valid git repository and dvc will fail, so tell the user
    echo "You have not initialised a git repo. Please run ./init_git.sh first and try again"
    exit
fi

if [ -z "$1" ]; then
    echo "Error: Missing required argument."
    echo "Usage: $0 pipeline name"
    exit 1
fi

PIPELINE_NAME="$1"
PIPELINE_DIR="pipelines/$PIPELINE_NAME"
PIPELINE_SRC_DIR="src/low_sim_pdbbind/pipelines/$PIPELINE_NAME"
PIPELINE_TESTS_DIR="tests/pipelines/$PIPELINE_NAME"

mkdir -p "$PIPELINE_DIR"
mkdir -p "$PIPELINE_SRC_DIR"
mkdir -p "$PIPELINE_TESTS_DIR"

cd "$PIPELINE_DIR"

echo "Initializing dvc repo"
"$_SCRIPT_DIR"/.venv/bin/python -m dvc init --subdir

echo "Opting out of anonymous aggregate usage analytics"
dvc config core.analytics false

echo "Setting up dvc remote storage bucket"
cat >.dvc/config <<-EOM
[core]
autostage = true
remote = storage

[hydra]
enabled = true
config_dir = ../../src/low_sim_pdbbind/pipelines/$PIPELINE_NAME/config
config_name = main

['remote "storage"']
url = s3://low_sim_pdbbind--$PIPELINE_NAME
EOM

echo "Setting up pipeline files"
cat >dvc.yaml <<-EOM
---
stages:
  demo:
    cmd: >
      python ../../src/low_sim_pdbbind/pipelines/$PIPELINE_NAME/stages/hello.py
    params:
      - hello
EOM

mkdir data
touch data/.gitkeep

popd                   # return to original directory
pushd "${_SCRIPT_DIR}" # cd to script directory

cd "$PIPELINE_SRC_DIR"

echo "Setting up source code template"
mkdir -p stages
cat >stages/hello.py <<-EOM
from pathlib import Path
from typing import Any, Dict

import dvc.api

from low_sim_pdbbind.utils.dir import get_pipeline_dir


def main(
    config: Dict[str, Any],
    pipeline_dir: Path,
) -> None:
    print(f"Hello, config['hello']['hello_who']!")


if __name__ == "__main__":
    config = dvc.api.params_show()
    pipeline_dir = get_pipeline_dir()
    main(config, pipeline_dir)
EOM

mkdir -p config
cat >config/main.yaml <<-EOM
defaults:
  - _self_
  # disable Hydra's logging if using Hydra CLI outside of DVC
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

# convenient settings if using Hydra CLI outside of DVC
hydra:
  output_subdir: null
  job:
    chdir: false

hello:
  hello_who: world
EOM

popd                   # return to original directory
pushd "${_SCRIPT_DIR}" # cd to script directory

cd "$PIPELINE_TESTS_DIR"

echo "Setting up sample tests"
cat >conftest.py <<-EOM
import shutil
from pathlib import Path
from typing import Any, Dict

import pytest

_TESTS_DIR = Path(__file__).parent

@pytest.fixture()
def quick_hello_config() -> Dict[str, Any]:
    return {"hello": {"hello_who": "you"}}

@pytest.fixture()
def quick_tmp_repo(tmp_path: Path) -> Path:
    tmp_repo = tmp_path / "repo"
    shutil.copytree(_TESTS_DIR / "tmp_repos" / "quick", tmp_repo)
    return tmp_repo
EOM

mkdir integration
cat >integration/test_stages.py <<-EOM
from pathlib import Path

import pytest
from contextlib_chdir import chdir

from low_sim_pdbbind.pipelines.$PIPELINE_NAME.stages.hello import (
    main as hello_main,
)


@pytest.mark.parametrize("config_name", ["quick_hello"])
def test_hello(config_name: str, quick_tmp_repo: Path, request: pytest.FixtureRequest) -> None:
    with chdir(quick_tmp_repo):
        quick_config = request.getfixturevalue(f"{config_name}_config")
        hello_main(quick_config, quick_tmp_repo)

        assert True  # replace for a real test!
EOM

mkdir -p tmp_repos/quick/.dvc
touch tmp_repos/quick/.dvc/.gitkeep

popd # return to original directory

git add "$PIPELINE_DIR"
git add "$PIPELINE_SRC_DIR"
git add "$PIPELINE_TESTS_DIR"

unset _SCRIPT_DIR
unset PIPELINE_NAME
unset PIPELINE_DIR
unset PIPELINE_SRC_DIR
unset PIPELINE_TEST_DIR
