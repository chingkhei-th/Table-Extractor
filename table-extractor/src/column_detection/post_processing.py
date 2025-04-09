"""Post-processing utilities for column detection."""


def validate_cell_structure(cell_coordinates):
    """Validate and fix cell structure issues.

    Args:
        cell_coordinates (list): List of cell coordinates.

    Returns:
        list: Validated and fixed cell coordinates.
    """
    if not cell_coordinates:
        return cell_coordinates

    # Make sure all rows have the same number of cells (columns)
    max_cells = max([row["cell_count"] for row in cell_coordinates])

    for row in cell_coordinates:
        if row["cell_count"] < max_cells:
            # Need to add placeholder cells
            current_count = row["cell_count"]
            needed_cells = max_cells - current_count

            # Add placeholder cells (just duplicate the last cell)
            if current_count > 0 and needed_cells > 0:
                last_cell = row["cells"][-1]
                for _ in range(needed_cells):
                    # Create a new cell right next to the last one
                    new_cell = {
                        "column": [
                            last_cell["column"][2],  # Last cell's right edge
                            last_cell["column"][1],  # Same y-coords
                            last_cell["column"][2]
                            + (
                                last_cell["column"][2] - last_cell["column"][0]
                            ),  # Same width
                            last_cell["column"][3],
                        ],
                        "cell": [
                            last_cell["cell"][2],  # Last cell's right edge
                            last_cell["cell"][1],  # Same y-coords
                            last_cell["cell"][2]
                            + (
                                last_cell["cell"][2] - last_cell["cell"][0]
                            ),  # Same width
                            last_cell["cell"][3],
                        ],
                    }
                    row["cells"].append(new_cell)

            row["cell_count"] = len(row["cells"])

    return cell_coordinates
