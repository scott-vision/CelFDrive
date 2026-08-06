"""Serialize CellClicker review tracks to and from the tracking XML schema.

Boxes are normalized YOLO ``x_center, y_center, width, height`` values in the
range 0--1, and image paths refer to source timepoint images.
"""

import os
import xml.etree.ElementTree as ET


DEFAULT_CLASSES = {
    0: "prophase",
    1: "earlyprometaphase",
    2: "prometaphase",
    3: "metaphase",
    4: "anaphase",
    5: "telophase",
}

DEFAULT_BOX_TYPES = {
    "original": "Box carried forward from the original CellClicker series annotation.",
    "otsu": "Box adjusted by Otsu thresholding.",
    "sam2": "Box adjusted by SAM2.",
    "tightened": "Box manually tightened by a user.",
}


def indent_xml(elem, level=0):
    """Apply deterministic two-space indentation to an XML element tree.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        Root or child element to format in place.
    level : int, default=0
        Current indentation depth.
    """
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def create_tracking_root(classes=None, box_types=None, metadata=None):
    """Create the root element for the versioned tracking XML structure.

    Parameters
    ----------
    classes : mapping[int, str] or None
        Class IDs and phase names. ``None`` selects the built-in six-phase map.
    box_types : mapping[str, str] or None
        Box variant identifiers and their user-facing descriptions.
    metadata : mapping[str, object] or None
        Scalar provenance values serialized under ``Metadata``.

    Returns
    -------
    xml.etree.ElementTree.Element
        A ``TrackingData`` root with metadata, classes, box types, and tracks.
    """
    # An empty mapping is meaningful when a caller intentionally has no entries;
    # only omitted values should receive the schema defaults.
    if classes is None:
        classes = DEFAULT_CLASSES
    if box_types is None:
        box_types = DEFAULT_BOX_TYPES
    if metadata is None:
        metadata = {}

    root = ET.Element("TrackingData")

    metadata_elem = ET.SubElement(root, "Metadata")
    for key, value in metadata.items():
        ET.SubElement(metadata_elem, key).text = str(value)

    classes_elem = ET.SubElement(root, "Classes")
    for class_id, class_name in classes.items():
        class_elem = ET.SubElement(classes_elem, "Class")
        class_elem.set("id", str(class_id))
        class_elem.set("name", str(class_name))

    box_types_elem = ET.SubElement(root, "BoxTypes")
    for box_type, description in box_types.items():
        box_type_elem = ET.SubElement(box_types_elem, "BoxType")
        box_type_elem.set("id", str(box_type))
        box_type_elem.set("description", str(description))

    ET.SubElement(root, "Tracks")
    return root


def append_track(root, track):
    """Append one in-memory track record to a tracking XML root.

    ``track`` contains source metadata and ordered timepoints. Each box must
    provide normalized YOLO centre coordinates in the range 0--1.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root created by :func:`create_tracking_root`.
    track : mapping
        Tracking record with ``track_id``, ``series_id``, ``source_path``, and
        a ``timepoints`` sequence.

    Returns
    -------
    xml.etree.ElementTree.Element
        The appended ``Track`` element.
    """
    tracks_elem = root.find("Tracks")
    if tracks_elem is None:
        tracks_elem = ET.SubElement(root, "Tracks")

    track_elem = ET.SubElement(tracks_elem, "Track")
    track_elem.set("id", str(track["track_id"]))
    track_elem.set("source_path", str(track["source_path"]))
    track_elem.set("series_id", str(track["series_id"]))
    track_elem.set("length", str(len(track["timepoints"])))

    for timepoint in track["timepoints"]:
        timepoint_elem = ET.SubElement(track_elem, "Timepoint")
        timepoint_elem.set("index", str(timepoint["timepoint_index"]))
        timepoint_elem.set("frame_index", str(timepoint["frame_index"]))
        timepoint_elem.set("image_path", str(timepoint["image_path"]))
        timepoint_elem.set("class_id", str(timepoint["class_id"]))
        timepoint_elem.set("phase", str(timepoint["phase_name"]))
        timepoint_elem.set("source_class_id", str(timepoint["source_class_id"]))
        preferred_box_type = timepoint.get("preferred_box_type")
        if preferred_box_type:
            timepoint_elem.set("preferred_box_type", str(preferred_box_type))

        boxes_elem = ET.SubElement(timepoint_elem, "Boxes")
        for box in timepoint["boxes"]:
            box_elem = ET.SubElement(boxes_elem, "Box")
            box_elem.set("type", str(box["box_type"]))
            box_elem.set("format", str(box.get("format", "yolo_xywh_norm")))
            box_elem.set("x_center", str(box["x_center"]))
            box_elem.set("y_center", str(box["y_center"]))
            box_elem.set("width", str(box["width"]))
            box_elem.set("height", str(box["height"]))
            if "source" in box:
                box_elem.set("source", str(box["source"]))

    return track_elem


def write_tracking_xml(output_file, tracks, classes=None, box_types=None, metadata=None):
    """Write tracking records as UTF-8 XML, creating a parent directory if set.

    Parameters
    ----------
    output_file : path-like
        Destination ``.xml`` path.
    tracks : sequence[mapping]
        Records accepted by :func:`append_track`.
    classes, box_types, metadata : mapping, optional
        Schema metadata written to the root element.
    """
    root = create_tracking_root(classes=classes, box_types=box_types, metadata=metadata)
    for track in tracks:
        append_track(root, track)

    indent_xml(root)
    output_directory = os.path.dirname(output_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def read_tracking_xml(xml_file):
    """Read tracking XML into dictionaries with numeric box coordinates.

    Parameters
    ----------
    xml_file : path-like
        Existing tracking XML document.

    Returns
    -------
    dict
        ``metadata``, ``classes``, ``box_types``, and ``tracks``. Box values
        are floats in normalized YOLO centre-box coordinates.

    Raises
    ------
    xml.etree.ElementTree.ParseError
        If the document is malformed.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    metadata = {}
    metadata_elem = root.find("Metadata")
    if metadata_elem is not None:
        for child in metadata_elem:
            metadata[child.tag] = child.text

    classes = {}
    classes_elem = root.find("Classes")
    if classes_elem is not None:
        for class_elem in classes_elem.findall("Class"):
            classes[int(class_elem.get("id"))] = class_elem.get("name")

    box_types = {}
    box_types_elem = root.find("BoxTypes")
    if box_types_elem is not None:
        for box_type_elem in box_types_elem.findall("BoxType"):
            box_types[box_type_elem.get("id")] = box_type_elem.get("description")

    tracks = []
    tracks_elem = root.find("Tracks")
    if tracks_elem is None:
        return {"metadata": metadata, "classes": classes, "box_types": box_types, "tracks": tracks}

    for track_elem in tracks_elem.findall("Track"):
        track = {
            "track_id": track_elem.get("id"),
            "source_path": track_elem.get("source_path"),
            "series_id": track_elem.get("series_id"),
            "timepoints": [],
        }
        for timepoint_elem in track_elem.findall("Timepoint"):
            timepoint = {
                "timepoint_index": int(timepoint_elem.get("index")),
                "frame_index": int(timepoint_elem.get("frame_index")),
                "image_path": timepoint_elem.get("image_path"),
                "class_id": int(timepoint_elem.get("class_id")),
                "phase_name": timepoint_elem.get("phase"),
                "source_class_id": int(timepoint_elem.get("source_class_id")),
                "preferred_box_type": timepoint_elem.get("preferred_box_type"),
                "boxes": [],
            }
            boxes_elem = timepoint_elem.find("Boxes")
            if boxes_elem is not None:
                for box_elem in boxes_elem.findall("Box"):
                    timepoint["boxes"].append(
                        {
                            "box_type": box_elem.get("type"),
                            "format": box_elem.get("format"),
                            "x_center": float(box_elem.get("x_center")),
                            "y_center": float(box_elem.get("y_center")),
                            "width": float(box_elem.get("width")),
                            "height": float(box_elem.get("height")),
                            "source": box_elem.get("source"),
                        }
                    )
            track["timepoints"].append(timepoint)
        tracks.append(track)

    return {"metadata": metadata, "classes": classes, "box_types": box_types, "tracks": tracks}


def write_tracking_data(output_file, tracking_data):
    """Write a dictionary returned by :func:`read_tracking_xml` back to XML."""
    write_tracking_xml(
        output_file=output_file,
        tracks=tracking_data.get("tracks", []),
        classes=tracking_data.get("classes"),
        box_types=tracking_data.get("box_types"),
        metadata=tracking_data.get("metadata"),
    )
