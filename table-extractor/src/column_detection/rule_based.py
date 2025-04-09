"""Rule-based approaches for column detection."""


def detect_columns_by_spacing(table_data, min_gap_ratio=0.02):
    """Detect columns based on spacing between words.

    Args:
        table_data (list): List of text elements with bounding boxes.
        min_gap_ratio (float): Minimum gap ratio to consider as column separator.

    Returns:
        list: List of detected column boundaries.
    """
    if not table_data:
        return []

    # Extract all x-coordinates
    x_coords = []
    for item in table_data:
        if "bbox" in item:
            bbox = item["bbox"]
            x_coords.append(bbox[0])  # left edge
            x_coords.append(bbox[2])  # right edge

    if not x_coords:
        return []

    # Calculate the width of the table
    table_width = max(x_coords) - min(x_coords)
    min_gap = table_width * min_gap_ratio

    # Sort coordinates and find large gaps
    x_coords.sort()
    gaps = []
    for i in range(1, len(x_coords)):
        gap = x_coords[i] - x_coords[i - 1]
        if gap > min_gap:
            gaps.append((x_coords[i - 1], x_coords[i], gap))

    # Sort gaps by size (largest first)
    gaps.sort(key=lambda x: x[2], reverse=True)

    # Take the top N gaps as column separators
    # This is a simplistic approach - would need more complex logic for real-world use
    columns = []
    if gaps:
        # Add the first column (from start to first gap)
        columns.append([min(x_coords), gaps[0][0]])

        # Add middle columns
        for i in range(len(gaps) - 1):
            columns.append([gaps[i][1], gaps[i + 1][0]])

        # Add the last column (from last gap to end)
        columns.append([gaps[-1][1], max(x_coords)])

    return columns
