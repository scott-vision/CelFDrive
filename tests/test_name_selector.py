import pytest

from CellClicker.name_selector import annotator_filename, available_annotator_names


@pytest.mark.parametrize("name", ["aggregated_tracking", "tracking_review"])
def test_annotator_filename_rejects_workflow_owned_names(name):
    with pytest.raises(ValueError, match="reserved"):
        annotator_filename(name)


def test_available_annotator_names_hides_workflow_owned_xml_files(tmp_path):
    for filename in ("alice.xml", "aggregated_tracking.xml", "tracking_review.xml"):
        (tmp_path / filename).touch()

    assert available_annotator_names(tmp_path) == ["alice"]
