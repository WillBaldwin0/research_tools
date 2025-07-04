import os
import numpy as np


def get_valid_set(folder, allset, suffix=None):
    import os 
    contents = [item for item in os.listdir(folder) if (os.path.isfile(os.path.join(folder, item)) and item[:13]=='valid_indices')]
    if len(contents) == 0:
        raise FileNotFoundError('no valid indcies file')
    elif len(contents) > 1:
        if suffix is None:
            raise ValueError('many valid indices, need suffix')
        else:
            filename = f'valid_indices_{suffix}.txt'
    else:
        filename = contents[0]
        
    with open(os.path.join(folder, filename)) as f:
        numbers = [int(line) for line in f.readlines()]
    trainset = []
    validset = []
    for i in range(len(allset)):
        allset[i].info['train_index'] = i
        if i in numbers:
            validset.append(allset[i])
        else:
            trainset.append(allset[i])
    return trainset, validset


def split_list_by_fraction(lst, fraction):
    length=len(lst)
    indices_train = np.random.choice(length, round(length*fraction), replace=False)
    train = []
    test = []
    for i, item in enumerate(lst):
        if i in indices_train:
            train.append(item)
        else:
            test.append(item)
    return train, test


def split_to_configtypes(lst, configkey='config_type'):
    cfg_types = {}
    for att in lst:
        if not att.info[configkey] in cfg_types.keys():
            cfg_types[att.info[configkey]] = [att]
        else:
            cfg_types[att.info[configkey]].append(att)
    return cfg_types


def compare_atoms_lists(al1, al2):
    def compare_atoms_objects(a1, a2):
        if len(a1) == len(a2) and np.allclose(a1.positions, a2.positions):
            return True
        else:
            return False
    
    intersection_list1 = []
    intersection_list2 = []
    al1_not_al2 = []
    for j, item in enumerate(al1):
        found = False
        for i, reference in enumerate(al2):
            if compare_atoms_objects(item, reference):
                intersection_list1.append(item)
                intersection_list2.append(reference)
                found = True
                break
        if not found:
            al1_not_al2.append(item)
    
    al2_not_al1 = []
    for j, item in enumerate(al2):
        found = False
        for i, reference in enumerate(al1):
            if compare_atoms_objects(item, reference):
                found = True
                break
        if not found:
            al2_not_al1.append(item)
    
    return intersection_list1, intersection_list2, al1_not_al2, al2_not_al1


def transfer_keys(al1, al2, keys, default_value=None, allow_new_items=False):
    intersection_list2, intersection_list1, al2_not_al1, al1_not_al2 = compare_atoms_lists(al2, al1)
    if len(al2_not_al1) > 0 and not allow_new_items:
        print('al2_not_al1:', len(al2_not_al1))
        raise ValueError('new items in al2')

    for inter1, inter2 in zip(intersection_list1, intersection_list2):
        for key in keys:
            inter2.info[key] = inter1.info[key]

    if default_value is not None:
        for item in al2_not_al1:
            for key in keys:
                item.info[key] = default_value
    
    return intersection_list2


from collections import defaultdict


def compare_atom_lists_ids(list1, list2):
    """
    Compare two lists of `ase.Atoms` objects based on their `info['unique_identifier']`.

    Returns:
    - intersection1: List of Atoms objects from list1 that have a match in list2.
    - intersection2: List of Atoms objects from list2 that have a match in list1.
    - complement1: List of Atoms objects in list1 that have no match in list2.
    - complement2: List of Atoms objects in list2 that have no match in list1.
    """
    
    # Helper function to group atoms by unique identifier
    def group_by_identifier(atom_list):
        grouped = defaultdict(list)
        for atom in atom_list:
            identifier = atom.info.get('unique_identifier', None)
            if identifier:
                grouped[identifier].append(atom)
        return grouped
    
    # Group both lists by identifier
    grouped1 = group_by_identifier(list1)
    grouped2 = group_by_identifier(list2)

    # Find intersection and complements
    intersection1, intersection2 = [], []
    complement1, complement2 = [], []
    
    all_identifiers = set(grouped1.keys()).union(grouped2.keys())

    for identifier in all_identifiers:
        if identifier in grouped1 and identifier in grouped2:
            intersection1.extend(grouped1[identifier])
            intersection2.extend(grouped2[identifier])
        elif identifier in grouped1:
            complement1.extend(grouped1[identifier])
        elif identifier in grouped2:
            complement2.extend(grouped2[identifier])

    return intersection1, intersection2, complement1, complement2


def find_duplicate_atoms(atom_list):
    """
    Identifies duplicate `ase.Atoms` objects based on their `info['unique_identifier']`.

    Returns:
    - A list of lists, where each sublist contains Atoms objects that share the same identifier.
    - A deduplicated version of the input list, keeping only the first occurrence of each identifier.
    """
    
    # Dictionary to store atoms grouped by unique identifier
    grouped = defaultdict(list)
    
    # List for the deduplicated version
    deduplicated_list = []
    seen_identifiers = set()
    
    for atom in atom_list:
        identifier = atom.info.get('unique_identifier', None)
        if identifier:
            grouped[identifier].append(atom)
            # Keep only the first occurrence in the deduplicated list
            if identifier not in seen_identifiers:
                deduplicated_list.append(atom)
                seen_identifiers.add(identifier)

    # Return only groups that contain duplicates and the deduplicated list
    duplicates = [group for group in grouped.values() if len(group) > 1]
    
    return duplicates, deduplicated_list


def find_duplicate_atoms_positions(atoms_list):
    """
    Identifies duplicate `ase.Atoms` objects based on their positions.

    Returns:
    - A list of lists, where each sublist contains Atoms objects that share the same positions.
    - A deduplicated version of the input list, keeping only the first occurrence of each set of positions.
    """

    # Dictionary to store atoms grouped by positions
    grouped = defaultdict(list)

    # List for the deduplicated version
    deduplicated_list = []
    seen_positions = set()

    for atoms in atoms_list:
        positions = atoms.get_positions()
        key = tuple(positions.flatten())
        grouped[key].append(atoms)
        # Keep only the first occurrence in the deduplicated list
        if key not in seen_positions:
            deduplicated_list.append(atoms)
            seen_positions.add(key)

    # Return only groups that contain duplicates and the deduplicated list
    duplicates = [group for group in grouped.values() if len(group) > 1]

    return duplicates, deduplicated_list