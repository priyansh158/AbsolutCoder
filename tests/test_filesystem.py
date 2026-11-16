"""Windows extended paths stay inside the repo root."""

import os

import pytest

from absolutcoder.agent import workspace_backend
from absolutcoder.shell import GuardedShellBackend, RepoFilesystemBackend, strip_extended_prefix


def test_strip_extended_prefix_keeps_drive_and_unc():
    assert strip_extended_prefix("\\\\?\\C:\\repo\\sub\\app.js") == "C:\\repo\\sub\\app.js"
    assert strip_extended_prefix("\\\\?\\UNC\\server\\share\\app.js") == "\\\\server\\share\\app.js"
    assert strip_extended_prefix("//?/C:/repo/sub/app.js") == "C:\\repo\\sub\\app.js"
    assert strip_extended_prefix("//?/UNC/server/share/app.js") == "\\\\server\\share\\app.js"
    assert strip_extended_prefix("/absolutTicTac/app.js") == "/absolutTicTac/app.js"


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length paths")
def test_extended_path_write_stays_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    routed = workspace_backend(repo)
    assert isinstance(routed.default, GuardedShellBackend)
    assert isinstance(routed.routes["/"], RepoFilesystemBackend)

    extended = "\\\\?\\" + str(repo.resolve()) + "\\absolutTicTac\\app.js"
    result = routed.write(extended, "app\n")
    written = repo / "absolutTicTac" / "app.js"
    assert result.error is None
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == "app\n"
    written.resolve().relative_to(repo.resolve())

    slash = "//?/" + repo.resolve().as_posix() + "/absolutTicTac/board.js"
    board = routed.write(slash, "board\n")
    assert board.error is None
    assert (repo / "absolutTicTac" / "board.js").read_text(encoding="utf-8") == "board\n"

    routed.write("/absolutTicTac/style.css", "css\n")
    assert (repo / "absolutTicTac" / "style.css").read_text(encoding="utf-8") == "css\n"

    outside = tmp_path / "outside.js"
    with pytest.raises(ValueError, match="outside root"):
        routed.write("\\\\?\\" + str(outside.resolve()), "no\n")
    assert not outside.exists()
    assert not (repo / "outside.js").exists()

    with pytest.raises(ValueError, match="outside root"):
        routed.write("\\\\?\\UNC\\evilhost\\share\\secret.txt", "no\n")
    with pytest.raises(ValueError, match="outside root"):
        routed.write("//?/UNC/evilhost/share/secret.txt", "no\n")
    assert list(repo.rglob("secret.txt")) == []
