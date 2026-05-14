from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


S_ascii_art_12 = """
___________
____o_o____
___o___o___
___oo______
______oo___
___o___o___
____o_o____
___________
"""

P_ascii_art_12 = """
___________
___o_o_____
___o___o___
___o___o___
___o_o_____
___o_______
__ooo______
___________
"""

Q_ascii_art_12 = """
___________
____o_o____
__o_____o__
__o_____o__
__o___o_o__
____o_o_o__
___________
"""

D_ascii_art_12 = """
___________
___o_o_____
___o__o____
___o___o___
___o___o___
___o__o____
___o_o_____
___________
"""

C_ascii_art_12 = """
__________
___ooo____
__o___o___
__o_______
__o_______
__o___o___
___ooo____
__________
"""

R_ascii_art_12 = """
___________
___o_o_____
___o___o___
___o___o___
___o_o_____
___o___o___
___o___o___
___________
"""

S_coordinate_map_12 = {}
P_coordinate_map_12 = {}
Q_coordinate_map_12 = {}
D_coordinate_map_12 = {}
C_coordinate_map_12 = {}
R_coordinate_map_12 = {}

# These are true 12-point mappings. The old 12-point mappings had labels up to
# 31, which made the 12-point coordinate maps incorrect.
S_key_mapping_12 = {4: 1, 2: 2, 1: 3, 3: 4, 5: 5, 6: 6, 7: 7, 8: 8, 10: 9, 12: 10, 11: 11, 9: 12}
P_key_mapping_12 = {10: 1, 11: 2, 12: 3, 9: 4, 7: 5, 5: 6, 3: 7, 1: 8, 2: 9, 4: 10, 6: 11, 8: 12}
Q_key_mapping_12 = {10: 1, 7: 2, 5: 3, 3: 4, 1: 5, 2: 6, 4: 7, 6: 8, 9: 9, 12: 10, 11: 11, 8: 12}
D_key_mapping_12 = {11: 1, 9: 2, 7: 3, 5: 4, 3: 5, 1: 6, 2: 7, 4: 8, 6: 9, 8: 10, 10: 11, 12: 12}
C_key_mapping_12 = {3: 1, 2: 2, 1: 3, 4: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 9: 11, 5: 12}
R_key_mapping_12 = {11: 1, 9: 2, 7: 3, 5: 4, 3: 5, 1: 6, 2: 7, 4: 8, 6: 9, 8: 10, 10: 11, 12: 12}

art_map_12 = [
    ["S", S_ascii_art_12, S_coordinate_map_12, S_key_mapping_12],
    ["P", P_ascii_art_12, P_coordinate_map_12, P_key_mapping_12],
    ["Q", Q_ascii_art_12, Q_coordinate_map_12, Q_key_mapping_12],
    ["D", D_ascii_art_12, D_coordinate_map_12, D_key_mapping_12],
    ["C", C_ascii_art_12, C_coordinate_map_12, C_key_mapping_12],
    ["R", R_ascii_art_12, R_coordinate_map_12, R_key_mapping_12],
]

S_ascii_art_31 = """
______________
______________
____oooooo____
___o______o___
___o______o___
___o__________
___o__________
____oooooo____
__________o___
__________o___
___oo_____o___
___o______o___
____oooooo____
______________
______________
"""

P_ascii_art_31 = """
______________
______________
___oooooooo___
___o_______o__
___o_______o__
___o_______o__
___o_______o__
___oooooooo___
___o__________
___o__________
___o__________
___o__________
__ooo_________
______________
______________
"""

Q_ascii_art_31 = """
_______________
_______________
_____ooooo_____
____o_____o____
___o_______o___
__o_________o__
__o_________o__
__o_________o__
__o_________o__
__o_________o__
___o_____o_o___
____o_____o____
_____ooooo_oo__
_______________
_______________
"""

D_ascii_art_31 = """
______________
______________
___ooooooo____
___o______o___
___o_______o__
___o_______o__
___o_______o__
___o_______o__
___o_______o__
___o_______o__
___o______o___
___o_____o____
___oooooo_____
______________
______________
"""

C_ascii_art_31 = """
______________
______________
___oooooooo___
__o________o__
__o________o__
__o________o__
__o___________
__o___________
__o___________
__o________o__
__o________o__
__o________o__
___oooooooo___
______________
______________
"""

R_ascii_art_31 = """
______________
______________
___ooooooo____
___o______o___
___o______o___
___o______o___
___o_____o____
___oooooo_____
___o_____o____
___o______o___
___o______o___
___o______o___
___o______o___
______________
______________
"""

S_coordinate_map_31 = {}
P_coordinate_map_31 = {}
Q_coordinate_map_31 = {}
D_coordinate_map_31 = {}
C_coordinate_map_31 = {}
R_coordinate_map_31 = {}

S_key_mapping_31 = {22: 1, 21: 2, 24: 3, 26: 4, 27: 5, 28: 6, 29: 7, 30: 8, 31: 9, 25: 10, 23: 11, 20: 12, 19: 13, 18: 14, 17: 15, 16: 16, 15: 17, 14: 18, 13: 19, 12: 20, 11: 21, 9: 22, 7: 23, 1: 24, 2: 25, 3: 26, 4: 27, 5: 28, 6: 29, 8: 30, 10: 31}
P_key_mapping_31 = {29: 1, 30: 2, 31: 3, 28: 4, 27: 5, 26: 6, 25: 7, 17: 8, 15: 9, 13: 10, 11: 11, 9: 12, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 10: 21, 12: 22, 14: 23, 16: 24, 24: 25, 23: 26, 22: 27, 21: 28, 20: 29, 19: 30, 18: 31}
Q_key_mapping_31 = {29: 1, 28: 2, 27: 3, 26: 4, 25: 5, 23: 6, 20: 7, 18: 8, 16: 9, 14: 10, 12: 11, 10: 12, 8: 13, 6: 14, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 7: 20, 9: 21, 11: 22, 13: 23, 15: 24, 17: 25, 19: 26, 22: 27, 21: 28, 24: 29, 30: 30, 31: 31}
D_key_mapping_31 = {26: 1, 24: 2, 22: 3, 20: 4, 18: 5, 16: 6, 14: 7, 12: 8, 10: 9, 8: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 9: 18, 11: 19, 13: 20, 15: 21, 17: 22, 19: 23, 21: 24, 23: 25, 25: 26, 31: 27, 30: 28, 29: 29, 28: 30, 27: 31}
C_key_mapping_31 = {19: 1, 21: 2, 23: 3, 31: 4, 30: 5, 29: 6, 28: 7, 27: 8, 26: 9, 25: 10, 24: 11, 22: 12, 20: 13, 18: 14, 17: 15, 16: 16, 15: 17, 13: 18, 11: 19, 9: 20, 1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26, 7: 27, 8: 28, 10: 29, 12: 30, 14: 31}
R_key_mapping_31 = {30: 1, 28: 2, 26: 3, 24: 4, 22: 5, 16: 6, 14: 7, 12: 8, 10: 9, 8: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 9: 18, 11: 19, 13: 20, 15: 21, 21: 22, 20: 23, 19: 24, 18: 25, 17: 26, 23: 27, 25: 28, 27: 29, 29: 30, 31: 31}

art_map_31 = [
    ["S", S_ascii_art_31, S_coordinate_map_31, S_key_mapping_31],
    ["P", P_ascii_art_31, P_coordinate_map_31, P_key_mapping_31],
    ["Q", Q_ascii_art_31, Q_coordinate_map_31, Q_key_mapping_31],
    ["D", D_ascii_art_31, D_coordinate_map_31, D_key_mapping_31],
    ["C", C_ascii_art_31, C_coordinate_map_31, C_key_mapping_31],
    ["R", R_ascii_art_31, R_coordinate_map_31, R_key_mapping_31],
]


def raw_coordinates_from_ascii(ascii_art: str) -> dict[int, list[int]]:
    """Read o characters from top-left to bottom-right and return raw coordinates."""
    lines = ascii_art.strip().splitlines()
    max_y = len(lines) - 1
    coordinate_map: dict[int, list[int]] = {}
    raw_label = 1

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == "o":
                coordinate_map[raw_label] = [x, max_y - y]
                raw_label += 1

    return coordinate_map


def apply_key_mapping(
    raw_coordinate_map: dict[int, list[int]],
    key_mapping: dict[int, int],
) -> dict[int, list[int]]:
    """Return only the remapped label -> coordinate entries."""
    expected_labels = set(range(1, len(raw_coordinate_map) + 1))
    mapped_labels = set(key_mapping.values())

    if mapped_labels != expected_labels:
        raise ValueError(
            f"Key mapping labels must be exactly 1-{len(raw_coordinate_map)}. "
            f"Got labels: {sorted(mapped_labels)}"
        )

    return {
        key_mapping[raw_label]: coordinate
        for raw_label, coordinate in raw_coordinate_map.items()
        if raw_label in key_mapping
    }


def build_coordinate_maps(art_map: list[list[object]]) -> None:
    for _letter, ascii_art, coordinate_map, key_mapping in art_map:
        raw_coordinate_map = raw_coordinates_from_ascii(str(ascii_art))
        coordinate_map.clear()
        coordinate_map.update(apply_key_mapping(raw_coordinate_map, key_mapping))


def coordinate_maps_to_dataframe(
    art_map: list[list[object]],
    number_of_points: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for letter, _ascii_art, coordinate_map, _key_mapping in art_map:
        for label, (x_axis, y_axis) in sorted(coordinate_map.items()):
            rows.append(
                {
                    "number_of_points": number_of_points,
                    "label": label,
                    "letter": letter,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                }
            )

    return pd.DataFrame(rows, columns=["number_of_points", "label", "letter", "x_axis", "y_axis"])


def save_scatter_plot(
    letter: str,
    coordinate_map: dict[int, list[int]],
    number_of_points: int,
    output_dir: str | Path,
    plot_bounds: tuple[float, float, float, float],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = sorted(coordinate_map)
    x_values = [coordinate_map[label][0] for label in labels]
    y_values = [coordinate_map[label][1] for label in labels]

    point_size = 1642 if number_of_points <= 12 else 638

    fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.scatter(
        x_values,
        y_values,
        s=point_size,
        c="#9a9a9a",
        edgecolors="white",
        linewidths=1.4,
        zorder=2,
    )

    x_min, x_max, y_min, y_max = plot_bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    image_path = output_path / f"{letter}_{number_of_points}_points.png"
    fig.savefig(image_path, facecolor="white")
    plt.close(fig)
    return image_path


def square_plot_bounds(
    art_map: list[list[object]],
    *,
    side_padding: float = 2.0,
    vertical_padding: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return shared square bounds for every scatter plot in one art map."""
    all_x_values: list[int] = []
    all_y_values: list[int] = []

    for _letter, _ascii_art, coordinate_map, _key_mapping in art_map:
        for x_axis, y_axis in coordinate_map.values():
            all_x_values.append(x_axis)
            all_y_values.append(y_axis)

    if not all_x_values or not all_y_values:
        raise ValueError("Cannot compute plot bounds without coordinates.")

    x_center = (min(all_x_values) + max(all_x_values)) / 2
    y_center = (min(all_y_values) + max(all_y_values)) / 2
    x_span = max(all_x_values) - min(all_x_values) + side_padding * 2
    y_span = max(all_y_values) - min(all_y_values) + vertical_padding * 2
    square_span = max(x_span, y_span)
    half_span = square_span / 2

    return (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
    )


def save_all_scatter_plots(
    art_map: list[list[object]],
    number_of_points: int,
    output_dir: str | Path,
) -> list[Path]:
    image_paths: list[Path] = []
    plot_bounds = square_plot_bounds(art_map)

    for letter, _ascii_art, coordinate_map, _key_mapping in art_map:
        image_paths.append(
            save_scatter_plot(
                str(letter),
                coordinate_map,
                number_of_points,
                output_dir,
                plot_bounds,
            )
        )

    return image_paths


def generate_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    build_coordinate_maps(art_map_12)
    build_coordinate_maps(art_map_31)

    dataframe_12 = coordinate_maps_to_dataframe(art_map_12, 12)
    dataframe_31 = coordinate_maps_to_dataframe(art_map_31, 31)

    dataframe_12.to_csv("letter_dataframe_12_points.csv", index=False)
    dataframe_31.to_csv("letter_dataframe_31_points.csv", index=False)

    save_all_scatter_plots(art_map_12, 12, "letter_images_12_points")
    save_all_scatter_plots(art_map_31, 31, "letter_images_31_points")

    return dataframe_12, dataframe_31


build_coordinate_maps(art_map_12)
build_coordinate_maps(art_map_31)


if __name__ == "__main__":
    df_12, df_31 = generate_outputs()
    print("Generated ascii_art_points_12.csv", df_12.shape)
    print("Generated ascii_art_points_31.csv", df_31.shape)
    print("Generated scatter images in letter_images_12_points and letter_images_31_points")
