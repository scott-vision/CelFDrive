"""Persist and read phase selections in CellClicker user XML files."""

import xml.etree.ElementTree as ET
import pandas as pd


def update_xml(file_name, series_id, selected_index, xml_file_name):
    """Upsert one legacy selected image index in a user XML document."""
    # Parse the existing XML file
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Search for an existing entry with the same "File" and "SeriesID"
    for data_entry in root.findall('DataEntry'):
        file_element = data_entry.find('PathName')
        series_id_element = data_entry.find('SeriesID')

        # Check if the file and series_id match
        if file_element is not None and series_id_element is not None and \
           file_element.text == file_name and series_id_element.text == str(series_id):
            # Update the selected_index value
            selected_index_element = data_entry.find('SelectedIndex')
            selected_index_element.text = str(selected_index)
            break
    else:
        # If no matching entry was found, create a new one
        data_entry = ET.Element('DataEntry')
        file_element = ET.SubElement(data_entry, 'PathName')
        series_id_element = ET.SubElement(data_entry, 'SeriesID')
        selected_index_element = ET.SubElement(data_entry, 'SelectedIndex')

        file_element.text = file_name
        series_id_element.text = str(series_id)
        selected_index_element.text = str(selected_index)

        root.append(data_entry)

    # Save the updated XML to the file
    tree.write(xml_file_name)

def update_xml_multiclass(file_name, series_id, selected_indices, xml_file_name, phases, selection_revision=0, phase_signature=None):
    """Upsert selected indices for ordered phase names in a user XML document."""
    # Parse the existing XML file
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Define the phases
    

    # Search for an existing entry with the same "File" and "SeriesID"
    for data_entry in root.findall('DataEntry'):
        file_element = data_entry.find('PathName')
        series_id_element = data_entry.find('SeriesID')

        # Check if the file and series_id match
        if file_element is not None and series_id_element is not None and \
           file_element.text == file_name and series_id_element.text == str(series_id):
            is_complete = all(phase in selected_indices for phase in phases)
            if is_complete:
                data_entry.set("selection_revision", str(selection_revision))
                if phase_signature:
                    data_entry.set("phase_signature", phase_signature)
            # Update the selected_index values for each phase
            for phase in phases:
                phase_element = data_entry.find(phase)
                phase_value = selected_indices.get(phase)
                if phase_value is None:
                    continue
                if phase_element is None:
                    phase_element = ET.SubElement(data_entry, phase)
                phase_element.text = str(-1 if phase_value in ['skipped', 'blurry'] else phase_value)
            break
    else:
        # If no matching entry was found, create a new one
        data_entry = ET.Element('DataEntry')
        if all(phase in selected_indices for phase in phases):
            data_entry.set("selection_revision", str(selection_revision))
            if phase_signature:
                data_entry.set("phase_signature", phase_signature)
        file_element = ET.SubElement(data_entry, 'PathName')
        series_id_element = ET.SubElement(data_entry, 'SeriesID')
        file_element.text = file_name
        series_id_element.text = str(series_id)

        # Create elements for each phase
        for phase in phases:
            phase_element = ET.SubElement(data_entry, phase)
            phase_value = selected_indices.get(phase)
            if phase_value is None:
                data_entry.remove(phase_element)
                continue
            phase_element.text = str(-1 if phase_value in ['skipped', 'blurry'] else phase_value)

        root.append(data_entry)

    # Save the updated XML to the file
    tree.write(xml_file_name)


def store_results(images_dict, selected_indicies, name_xml):
    """Persist one selected index for every image-series mapping entry."""

    for (file_name, series_id ), selected_index in zip(images_dict.keys(), selected_indicies):
        print(file_name, series_id, selected_index, name_xml)
        update_xml(file_name, series_id, selected_index, name_xml)    

def store_results_multiclass(images_dict, selected_indicies, name_xml, phases, revisions=None, phase_signature=None):
    """Persist phase selections for every image-series entry in ``images_dict``."""

    revisions = revisions or {}
    for (file_name, series_id ), selected_index_dict in zip(images_dict.keys(), selected_indicies):
        print(file_name, series_id, selected_index_dict, name_xml)
        update_xml_multiclass(
            file_name, series_id, selected_index_dict, name_xml, phases,
            selection_revision=revisions.get((file_name, str(series_id)), 0),
            phase_signature=phase_signature,
        )


def read_xml_to_dataframe(xml_file):
    """Return user XML selection records as a pandas dataframe."""
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Create a list to hold the data
    data_entries = []

    # Iterate through each 'DataEntry' in the XML
    for data_entry in root.findall('DataEntry'):
        file = data_entry.find('PathName').text
        print(file)
        series_id = data_entry.find('SeriesID').text
        selected_index = data_entry.find('SelectedIndex').text

        # Append this entry as a dict to the list
        data_entries.append({
            'PathName': file,
            'SeriesID': int(series_id),
            'SelectedIndex': int(selected_index)
        })

    # Create a DataFrame from the list of dicts
    df = pd.DataFrame(data_entries)
    return df


