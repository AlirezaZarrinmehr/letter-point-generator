"""
Create evenly spaced points along the centerline of an actual font letter.

Requires Pillow:
    pip install pillow

Main function:
    points, image = font_letter_points("S", 12, r"C:\\Windows\\Fonts\\arial.ttf")

The returned points use the same pixel coordinate system as the returned image:
    x increases to the right, y increases downward.
"""

from __future__ import annotations

import heapq
import math
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError("font_letter_points.py requires Pillow. Install it with: pip install pillow") from exc


Point = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class LetterPointResult:
    points: list[Point]
    image: Image.Image


def font_letter_points(
    letter: str,
    number_of_points: int,
    font: str | Path,
    *,
    font_size: int = 256,
    padding: int = 4,
    threshold: int = 32,
    smooth_iterations: int = 2,
    include_endpoints: bool = True,
    path_coverage_threshold: float = 0.55,
    loop_coverage_threshold: float = 0.25,
    circle_radius: float | None = None,
    circle_fit_scale: float = 0.92,
) -> tuple[list[Point], Image.Image]:
    """
    Return evenly spaced centerline points and an image of the actual font letter.

    Args:
        letter: Single letter or glyph to render, for example "S".
        number_of_points: Number of x/y points to return.
        font: Path to a TrueType/OpenType font file, for example
            r"C:\\Windows\\Fonts\\arial.ttf".
        font_size: Render size in pixels. Higher values give a smoother skeleton.
        padding: Transparent padding around the rendered glyph.
        threshold: Alpha threshold used to convert the rendered glyph into a mask.
        smooth_iterations: Number of Chaikin smoothing passes on the skeleton path.
        include_endpoints: If True, first and last points are the path endpoints.
        path_coverage_threshold: If the best ordered path covers less than this
            fraction of the skeleton, sample the whole skeleton graph instead.
            This matters for letters like Q, O, A, P, D, etc. that contain loops.
        loop_coverage_threshold: Very low path coverage usually means a
            loop-dominant glyph such as Q/O, which needs ring-style sampling.
        circle_radius: Optional radius, in pixels, that must fit inside the
            actual glyph at every returned coordinate. If omitted, it is
            estimated from the letter's stroke thickness.
        circle_fit_scale: Safety multiplier for the estimated radius. Values
            below 1 avoid anti-aliased edge leakage.

    Returns:
        (points, image)

        points is a list of (x, y) pixel coordinates aligned to image.
        image is a transparent RGBA image of the actual font-rendered letter.
    """
    if not letter:
        raise ValueError("letter must not be empty.")
    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")

    image, mask = render_font_letter(letter, font, font_size=font_size, padding=padding)
    binary = mask_to_binary(mask, threshold=threshold)
    skeleton = zhang_suen_thinning(binary)
    distance_map = distance_to_background(binary)
    pixels = largest_component(skeleton_pixels(skeleton))
    raw_path = longest_skeleton_path(skeleton)
    raw_path_coverage = len(set(raw_path)) / len(pixels) if pixels else 0.0
    endpoint_count = sum(1 for pixel in pixels if pixel_degree(pixel, pixels) == 1)

    if len(raw_path) < 2 or not pixels:
        raise ValueError("Could not find a centerline path for this glyph.")

    fit_radius = circle_radius
    if fit_radius is None:
        fit_radius = estimate_stroke_radius(distance_map, pixels) * circle_fit_scale

    safe_pixels = safe_skeleton_pixels(pixels, distance_map, fit_radius)
    if len(safe_pixels) < max(2, min(number_of_points, len(pixels) // 10)):
        safe_pixels = pixels

    if raw_path_coverage >= path_coverage_threshold and endpoint_count == 2:
        path_points = [(float(x), float(y)) for x, y in raw_path]
        path_points = smooth_polyline(path_points, iterations=smooth_iterations)
        points = sample_polyline(path_points, number_of_points, include_endpoints=include_endpoints)
    elif raw_path_coverage <= loop_coverage_threshold:
        points = even_graph_points(safe_pixels, number_of_points)
    else:
        points = even_graph_points(safe_pixels, number_of_points)

    points = snap_points_to_safe_region(points, safe_pixels, distance_map, fit_radius)
    if len(letter) == 1 and letter.lower() in {"p", "r"}:
        points = order_stemmed_letter_labels(points, letter)
    return points, image


def letter_points_dataframe(
    number_of_points: int,
    font: str | Path,
    *,
    letters: str = string.ascii_uppercase,
    output_dir: str | Path = "letter_images",
    font_size: int = 256,
    padding: int = 4,
    threshold: int = 32,
    round_digits: int | None = 4,
    coordinate_system: str = "cartesian",
    save_point_overlay: bool = False,
    fixed_canvas: bool = True,
    canvas_size: tuple[int, int] | None = None,
    background_fill: tuple[int, int, int, int] | None = (255, 255, 255, 255),
    include_alignment_columns: bool = False,
    include_picture_path: bool = False,
) -> "pd.DataFrame":
    """
    Build a DataFrame of sampled points for letters A-Z.

    By default the returned DataFrame has these columns:
        number_of_points, label, letter, x_axis, y_axis

    Example:
        df = letter_points_dataframe(12, r"C:\\Windows\\Fonts\\arial.ttf")

    Args:
        number_of_points: Number of points to generate for each letter.
        font: Path to the font file.
        letters: Letters to include. Default is uppercase A-Z.
        output_dir: Folder where letter images are saved.
        font_size: Render size in pixels.
        padding: Transparent padding around each rendered letter.
        threshold: Alpha threshold used for the glyph mask.
        round_digits: Decimal places for x/y values. Use None to keep raw floats.
        coordinate_system: "cartesian" returns plot-ready coordinates with y
            increasing upward. "image" returns image pixel coordinates with y
            increasing downward.
        save_point_overlay: If True, saved images include the generated points.
            If False, saved images contain the plain actual letter from the font.
        fixed_canvas: If True, every saved letter image uses the same canvas
            size and the points are shifted to that canvas. This avoids
            background-image rescaling drift in visualization tools.
        canvas_size: Optional (width, height) for the fixed canvas. If omitted,
            the largest rendered letter image in this export is used.
        background_fill: RGBA fill for the exported image background. The
            default white background prevents tools from trimming transparent
            padding. Use None to keep transparent PNGs.
        include_alignment_columns: If True, include image bounds and size
            columns so visualization tools can place each picture from
            image_x_min/image_y_min to image_x_max/image_y_max.
        include_picture_path: If True, include the saved image path in the
            returned DataFrame.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("letter_points_dataframe requires pandas. Install it with: pip install pandas") from exc

    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")
    if coordinate_system not in {"cartesian", "image"}:
        raise ValueError('coordinate_system must be "cartesian" or "image".')
    if canvas_size is not None:
        canvas_width, canvas_height = canvas_size
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("canvas_size must contain positive width and height.")

    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, list[Point], Image.Image]] = []
    for letter in letters:
        points, image = font_letter_points(
            letter,
            number_of_points,
            font,
            font_size=font_size,
            padding=padding,
            threshold=threshold,
        )
        generated.append((letter, points, image))

    if fixed_canvas:
        if canvas_size is None:
            canvas_width = max(image.width for _, _, image in generated)
            canvas_height = max(image.height for _, _, image in generated)
        else:
            canvas_width, canvas_height = canvas_size
    else:
        canvas_width = canvas_height = 0

    rows: list[dict[str, object]] = []

    for letter, points, image in generated:
        if fixed_canvas:
            if image.width > canvas_width or image.height > canvas_height:
                raise ValueError(
                    f"canvas_size {canvas_size!r} is smaller than rendered letter {letter!r} "
                    f"({image.width}x{image.height})."
                )
            offset_x = (canvas_width - image.width) // 2
            offset_y = (canvas_height - image.height) // 2
            export_image = place_image_on_canvas(
                image,
                (canvas_width, canvas_height),
                (offset_x, offset_y),
                background_fill=background_fill,
            )
            export_points = [(x + offset_x, y + offset_y) for x, y in points]
        else:
            export_image = apply_background_fill(image, background_fill)
            export_points = points

        picture_path = image_dir / f"{letter}_{number_of_points}_points.png"
        if save_point_overlay:
            draw_points_overlay(export_image, export_points).save(picture_path)
        else:
            export_image.save(picture_path)

        image_width = export_image.width
        image_height = export_image.height

        for label, (x_axis, y_axis) in enumerate(export_points, start=1):
            if coordinate_system == "cartesian":
                y_axis = image_height - y_axis

            if round_digits is not None:
                x_axis = round(x_axis, round_digits)
                y_axis = round(y_axis, round_digits)

            row: dict[str, object] = {
                "number_of_points": number_of_points,
                "label": label,
                "letter": letter,
                "x_axis": x_axis,
                "y_axis": y_axis,
            }
            if include_alignment_columns:
                row.update(
                    {
                        "image_x_min": 0,
                        "image_x_max": image_width,
                        "image_y_min": 0,
                        "image_y_max": image_height,
                        "image_width": image_width,
                        "image_height": image_height,
                    }
                )
            if include_picture_path:
                row["picture_path"] = str(picture_path.resolve())
            rows.append(row)

    columns = [
        "number_of_points",
        "label",
        "letter",
        "x_axis",
        "y_axis",
    ]
    if include_alignment_columns:
        columns.extend(
            [
                "image_x_min",
                "image_x_max",
                "image_y_min",
                "image_y_max",
                "image_width",
                "image_height",
            ]
        )
    if include_picture_path:
        columns.append("picture_path")

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def place_image_on_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    offset: tuple[int, int],
    *,
    background_fill: tuple[int, int, int, int] | None = (255, 255, 255, 255),
) -> Image.Image:
    """Place an RGBA image on a fixed canvas while preserving its pixel frame."""
    fill = background_fill if background_fill is not None else (255, 255, 255, 0)
    canvas = Image.new("RGBA", canvas_size, fill)
    canvas.alpha_composite(image.convert("RGBA"), offset)
    return canvas


def apply_background_fill(
    image: Image.Image,
    background_fill: tuple[int, int, int, int] | None = (255, 255, 255, 255),
) -> Image.Image:
    """Apply a solid background to an RGBA image, or keep it transparent."""
    if background_fill is None:
        return image.copy().convert("RGBA")

    canvas = Image.new("RGBA", image.size, background_fill)
    canvas.alpha_composite(image.convert("RGBA"), (0, 0))
    return canvas


def render_font_letter(
    letter: str,
    font: str | Path,
    *,
    font_size: int = 256,
    padding: int = 4,
    fill: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> tuple[Image.Image, Image.Image]:
    """Render the actual font glyph into an RGBA image and an alpha mask."""
    font_obj = ImageFont.truetype(str(font), font_size)

    scratch = Image.new("L", (1, 1), 0)
    scratch_draw = ImageDraw.Draw(scratch)
    left, top, right, bottom = scratch_draw.textbbox((0, 0), letter, font=font_obj)

    width = max(1, right - left + padding * 2)
    height = max(1, bottom - top + padding * 2)
    draw_at = (padding - left, padding - top)

    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    mask = Image.new("L", (width, height), 0)

    ImageDraw.Draw(image).text(draw_at, letter, font=font_obj, fill=fill)
    ImageDraw.Draw(mask).text(draw_at, letter, font=font_obj, fill=255)
    return image, mask


def mask_to_binary(mask: Image.Image, *, threshold: int = 32) -> list[bytearray]:
    """Convert a Pillow L image to a foreground/background grid."""
    gray = mask.convert("L")
    width, height = gray.size
    pixels = gray.tobytes()
    return [
        bytearray(1 if pixels[y * width + x] > threshold else 0 for x in range(width))
        for y in range(height)
    ]


def distance_to_background(binary: list[bytearray]) -> list[list[float]]:
    """Approximate Euclidean distance from each pixel to the glyph boundary."""
    height = len(binary)
    width = len(binary[0]) if height else 0
    distances = [[math.inf for _ in range(width)] for _ in range(height)]
    heap: list[tuple[float, int, int]] = []

    for y, row in enumerate(binary):
        for x, value in enumerate(row):
            if not value:
                distances[y][x] = 0.0
                heapq.heappush(heap, (0.0, x, y))

    steps = [
        (-1, -1, math.sqrt(2)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2)),
    ]

    while heap:
        current_distance, x, y = heapq.heappop(heap)
        if current_distance != distances[y][x]:
            continue

        for dx, dy, step in steps:
            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            next_distance = current_distance + step
            if next_distance < distances[ny][nx]:
                distances[ny][nx] = next_distance
                heapq.heappush(heap, (next_distance, nx, ny))

    return distances


def estimate_stroke_radius(distance_map: list[list[float]], pixels: set[Pixel]) -> float:
    """
    Estimate half of the usable stroke thickness from skeleton distances.

    The lower-middle percentile avoids endpoints and junction artifacts while
    still respecting thinner strokes such as the leg of R or tail of Q.
    """
    values = sorted(
        distance_map[y][x]
        for x, y in pixels
        if 0 <= y < len(distance_map)
        and 0 <= x < len(distance_map[y])
        and math.isfinite(distance_map[y][x])
        and distance_map[y][x] > 0
    )

    if not values:
        return 1.0

    return max(1.0, values[int((len(values) - 1) * 0.30)])


def safe_skeleton_pixels(
    pixels: set[Pixel],
    distance_map: list[list[float]],
    radius: float,
) -> set[Pixel]:
    """Keep skeleton pixels where a circle of radius fits inside the glyph."""
    return {
        (x, y)
        for x, y in pixels
        if 0 <= y < len(distance_map)
        and 0 <= x < len(distance_map[y])
        and distance_map[y][x] >= radius
    }


def pixels_to_grid(pixels: set[Pixel], width: int, height: int) -> list[bytearray]:
    grid = [bytearray(width) for _ in range(height)]

    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 1

    return grid


def snap_points_to_safe_region(
    points: list[Point],
    safe_pixels: set[Pixel],
    distance_map: list[list[float]],
    radius: float,
) -> list[Point]:
    """Snap continuous samples to nearby circle-safe medial-axis pixels."""
    if not safe_pixels:
        return points

    snapped: list[Point] = []
    used: set[Pixel] = set()

    for point in points:
        alternatives = sorted(
            safe_pixels,
            key=lambda pixel: (
                math.hypot(pixel[0] - point[0], pixel[1] - point[1]),
                -distance_map[pixel[1]][pixel[0]],
            ),
        )
        best = alternatives[0]

        for alternative in alternatives:
            if alternative in used:
                continue
            best = alternative
            break

        used.add(best)
        snapped.append((float(best[0]), float(best[1])))

    return snapped


def zhang_suen_thinning(binary: list[bytearray]) -> list[bytearray]:
    """Skeletonize a binary image using the Zhang-Suen thinning algorithm."""
    grid = [bytearray(row) for row in binary]
    height = len(grid)
    width = len(grid[0]) if height else 0
    changed = True

    while changed:
        changed = False
        changed = thinning_pass(grid, width, height, pass_index=0) or changed
        changed = thinning_pass(grid, width, height, pass_index=1) or changed

    return grid


def thinning_pass(grid: list[bytearray], width: int, height: int, *, pass_index: int) -> bool:
    to_remove: list[Pixel] = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not grid[y][x]:
                continue

            n = neighbors_clockwise(grid, x, y)
            count = sum(n)
            transitions = zero_to_one_transitions(n)

            if count < 2 or count > 6 or transitions != 1:
                continue

            p2, _, p4, _, p6, _, p8, _ = n

            if pass_index == 0:
                if p2 and p4 and p6:
                    continue
                if p4 and p6 and p8:
                    continue
            else:
                if p2 and p4 and p8:
                    continue
                if p2 and p6 and p8:
                    continue

            to_remove.append((x, y))

    for x, y in to_remove:
        grid[y][x] = 0

    return bool(to_remove)


def neighbors_clockwise(grid: list[bytearray], x: int, y: int) -> list[int]:
    # P2, P3, P4, P5, P6, P7, P8, P9 in Zhang-Suen notation.
    return [
        grid[y - 1][x],
        grid[y - 1][x + 1],
        grid[y][x + 1],
        grid[y + 1][x + 1],
        grid[y + 1][x],
        grid[y + 1][x - 1],
        grid[y][x - 1],
        grid[y - 1][x - 1],
    ]


def zero_to_one_transitions(values: list[int]) -> int:
    return sum(1 for i, value in enumerate(values) if not value and values[(i + 1) % len(values)])


def longest_skeleton_path(skeleton: list[bytearray]) -> list[Pixel]:
    """Find the longest endpoint-to-endpoint path in the largest skeleton component."""
    pixels = largest_component(skeleton_pixels(skeleton))
    if len(pixels) < 2:
        return list(pixels)

    endpoints = [pixel for pixel in pixels if pixel_degree(pixel, pixels) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(pixels)

    # If the skeleton has many tiny spurs, try the most extreme endpoints first.
    candidates = sorted(
        candidates,
        key=lambda p: (p[1], p[0], -p[1], -p[0]),
    )

    best_distance = -1.0
    best_path: list[Pixel] = []

    for start in candidates:
        distances, previous = dijkstra_skeleton(pixels, start)
        for end in candidates:
            distance_value = distances.get(end)
            if distance_value is not None and distance_value > best_distance:
                best_distance = distance_value
                best_path = rebuild_path(previous, start, end)

    return orient_top_to_bottom(best_path)


def sample_skeleton_graph(pixels: set[Pixel], number_of_points: int) -> list[Point]:
    """
    Evenly sample an entire skeleton graph.

    This is used when a glyph skeleton is not a single line. For example, Q has
    a loop plus a tail, so a single longest path captures only a slice of the
    letter. The sampler handles that as a loop plus any open branches and spaces
    points by arc length across those pieces.
    """
    if not pixels:
        raise ValueError("Cannot sample an empty skeleton graph.")
    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")

    loop_points = sample_loop_plus_tail(pixels, number_of_points)
    if loop_points:
        return loop_points

    return farthest_graph_points(pixels, number_of_points)


def sample_loop_plus_tail(pixels: set[Pixel], number_of_points: int) -> list[Point]:
    """Sample looped glyphs such as Q/O by ring arc length plus tail length."""
    if len(pixels) < 20:
        return []

    center = bounding_box_center(pixels)
    radii = sorted(math.hypot(x - center[0], y - center[1]) for x, y in pixels)
    median_radius = radii[len(radii) // 2]
    ring_radius_limit = radii[min(len(radii) - 1, int(len(radii) * 0.90))]
    ring_pixels = {
        pixel
        for pixel in pixels
        if math.hypot(pixel[0] - center[0], pixel[1] - center[1]) <= ring_radius_limit
    }

    if len(ring_pixels) < 10:
        return []

    ring_path = ring_path_by_angle(ring_pixels, center, median_radius)
    if len(ring_path) < 4:
        return []

    closed_ring = [*ring_path, ring_path[0]]
    tail_path = endpoint_diameter_path(pixels)
    tail_length = polyline_length([(float(x), float(y)) for x, y in tail_path])
    ring_length = polyline_length(closed_ring)
    has_tail = tail_length > median_radius * 0.35
    tail_count = 0

    if has_tail and number_of_points >= 4:
        tail_count = round(number_of_points * tail_length / (ring_length + tail_length))
        tail_count = max(1, min(tail_count, max(1, number_of_points // 3)))

    ring_count = number_of_points - tail_count
    points = sample_polyline(closed_ring, ring_count, include_endpoints=False)

    if tail_count:
        tail_points = sample_polyline(
            [(float(x), float(y)) for x, y in tail_path],
            tail_count,
            include_endpoints=True,
        )
        points.extend(tail_points)

    return points


def bounding_box_center(pixels: set[Pixel]) -> Point:
    min_x = min(x for x, _ in pixels)
    max_x = max(x for x, _ in pixels)
    min_y = min(y for _, y in pixels)
    max_y = max(y for _, y in pixels)
    return ((min_x + max_x) / 2, (min_y + max_y) / 2)


def ring_path_by_angle(ring_pixels: set[Pixel], center: Point, median_radius: float) -> list[Point]:
    start = min(ring_pixels, key=lambda pixel: (pixel[1], pixel[0]))
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    ordered: list[Pixel] = []
    used: set[Pixel] = set()
    angle_samples = min(720, max(90, len(ring_pixels) * 2))

    for i in range(angle_samples):
        target_angle = start_angle + math.tau * i / angle_samples
        pixel = min(
            ring_pixels,
            key=lambda candidate: ring_pixel_score(candidate, center, target_angle, median_radius),
        )

        if pixel not in used:
            ordered.append(pixel)
            used.add(pixel)

    if len(ordered) < 4:
        return []

    return smooth_polyline([(float(x), float(y)) for x, y in ordered], iterations=1)


def ring_pixel_score(pixel: Pixel, center: Point, target_angle: float, median_radius: float) -> float:
    angle = math.atan2(pixel[1] - center[1], pixel[0] - center[0])
    radius = math.hypot(pixel[0] - center[0], pixel[1] - center[1])
    return angular_distance(angle, target_angle) * median_radius + abs(radius - median_radius) * 0.2


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % math.tau - math.pi)


def endpoint_diameter_path(pixels: set[Pixel]) -> list[Pixel]:
    endpoints = [pixel for pixel in pixels if pixel_degree(pixel, pixels) == 1]
    if len(endpoints) < 2:
        return []

    best_distance = -1.0
    best_path: list[Pixel] = []

    for start in endpoints:
        distances, previous = dijkstra_skeleton(pixels, start)
        for end in endpoints:
            distance_value = distances.get(end)
            if distance_value is not None and distance_value > best_distance:
                best_distance = distance_value
                best_path = rebuild_path(previous, start, end)

    return best_path


def polyline_length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def skeleton_core(pixels: set[Pixel]) -> set[Pixel]:
    """Return the 2-core of the skeleton graph, which preserves loop pixels."""
    neighbors = {pixel: set(graph_neighbors(pixel, pixels)) for pixel in pixels}
    degree = {pixel: len(pixel_neighbors) for pixel, pixel_neighbors in neighbors.items()}
    core = set(pixels)
    queue = [pixel for pixel, value in degree.items() if value <= 1]
    index = 0

    while index < len(queue):
        pixel = queue[index]
        index += 1

        if pixel not in core:
            continue

        core.remove(pixel)

        for neighbor in neighbors[pixel]:
            if neighbor not in core:
                continue

            degree[neighbor] -= 1
            if degree[neighbor] <= 1:
                queue.append(neighbor)

    return core


def order_loop_by_angle(core: set[Pixel]) -> list[Point]:
    """
    Order loop pixels around their centroid.

    Skeleton loops from raster thinning often contain small junction clusters, so
    a strict graph walk is brittle. Angle ordering produces a stable closed
    center contour for looped glyphs such as Q and O.
    """
    center_x = sum(x for x, _ in core) / len(core)
    center_y = sum(y for _, y in core) / len(core)
    ordered = sorted(
        core,
        key=lambda pixel: math.atan2(pixel[1] - center_y, pixel[0] - center_x),
    )
    start_index = min(range(len(ordered)), key=lambda index: (ordered[index][1], ordered[index][0]))
    ordered = ordered[start_index:] + ordered[:start_index]
    return smooth_polyline([(float(x), float(y)) for x, y in ordered], iterations=1)


def tail_paths_from_core(pixels: set[Pixel], core: set[Pixel]) -> list[list[Point]]:
    """Find open branch paths attached to a loop core, such as the tail of Q."""
    branch_pixels = pixels - core
    components = connected_components(branch_pixels)
    paths: list[list[Point]] = []

    for component in components:
        attached = [
            pixel
            for pixel in component
            if any(neighbor in core for neighbor in adjacent_pixels(pixel))
        ]
        if not attached:
            continue

        endpoint_candidates = [
            pixel
            for pixel in component
            if pixel_degree(pixel, component) <= 1
        ] or list(component)
        best_path: list[Pixel] = []
        best_distance = -1.0

        for start in attached:
            distances, previous = dijkstra_skeleton(component, start)
            for end in endpoint_candidates:
                distance_value = distances.get(end)
                if distance_value is not None and distance_value > best_distance:
                    best_distance = distance_value
                    best_path = rebuild_path(previous, start, end)

        if len(best_path) < 2:
            continue

        core_start = nearest_core_neighbor(best_path[0], core)
        if core_start is not None:
            best_path = [core_start, *best_path]

        paths.append([(float(x), float(y)) for x, y in best_path])

    return paths


def connected_components(pixels: set[Pixel]) -> list[set[Pixel]]:
    remaining = set(pixels)
    components: list[set[Pixel]] = []

    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]

        while stack:
            pixel = stack.pop()
            for neighbor in list(graph_neighbors(pixel, remaining)):
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)

        components.append(component)

    return components


def nearest_core_neighbor(pixel: Pixel, core: set[Pixel]) -> Pixel | None:
    candidates = [neighbor for neighbor in adjacent_pixels(pixel) if neighbor in core]
    if not candidates:
        return None

    return min(candidates, key=lambda neighbor: math.hypot(neighbor[0] - pixel[0], neighbor[1] - pixel[1]))


def adjacent_pixels(pixel: Pixel) -> Iterable[Pixel]:
    x, y = pixel

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            yield (x + dx, y + dy)


def sample_polyline_pieces(pieces: list[list[Point]], number_of_points: int) -> list[Point]:
    """Sample a collection of polylines by their combined arc length."""
    edges: list[tuple[Point, Point, float]] = []

    for piece in pieces:
        for a, b in zip(piece, piece[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length > 0:
                edges.append((a, b, length))

    total_length = sum(length for _, _, length in edges)
    if total_length <= 0:
        raise ValueError("Skeleton graph has no measurable length.")

    sampled: list[Point] = []
    edge_index = 0
    edge_start_distance = 0.0

    for i in range(number_of_points):
        target = total_length * i / number_of_points

        while edge_index < len(edges) - 1 and edge_start_distance + edges[edge_index][2] < target:
            edge_start_distance += edges[edge_index][2]
            edge_index += 1

        a, b, length = edges[edge_index]
        t = 0.0 if length == 0 else (target - edge_start_distance) / length
        sampled.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

    return sampled


def farthest_graph_points(pixels: set[Pixel], number_of_points: int) -> list[Point]:
    """Fallback sampler for complex graphs that do not expose a clean loop."""
    selected = [topmost_pixel(pixels)]
    min_distances = {pixel: math.inf for pixel in pixels}

    while len(selected) < number_of_points:
        distances, _ = dijkstra_skeleton(pixels, selected[-1])

        for pixel, distance_value in distances.items():
            if distance_value < min_distances[pixel]:
                min_distances[pixel] = distance_value

        next_pixel = max(
            pixels,
            key=lambda pixel: (
                min_distances[pixel],
                abs(pixel[0] - selected[0][0]) + abs(pixel[1] - selected[0][1]),
                -pixel[1],
                -pixel[0],
            ),
        )

        if next_pixel in selected:
            break

        selected.append(next_pixel)

    selected = order_points_for_display(selected)
    return [(float(x), float(y)) for x, y in selected]


def even_graph_points(pixels: set[Pixel], number_of_points: int, *, iterations: int = 8) -> list[Point]:
    """
    Place points evenly over a skeleton network.

    This is for branched letters such as P and R, where there is no single
    natural stroke path. It treats skeleton pixels as uniformly weighted line
    samples, then relaxes point centers so each point owns a similar amount of
    the letter-line network.
    """
    if not pixels:
        raise ValueError("Cannot sample an empty skeleton graph.")
    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")

    candidates = list(pixels)
    selected = [topmost_pixel(pixels)]

    while len(selected) < min(number_of_points, len(candidates)):
        selected.append(
            max(
                candidates,
                key=lambda pixel: min(squared_distance(pixel, chosen) for chosen in selected),
            ),
        )

    for _ in range(iterations):
        clusters: list[list[Pixel]] = [[] for _ in selected]

        for pixel in candidates:
            cluster_index = min(
                range(len(selected)),
                key=lambda index: squared_distance(pixel, selected[index]),
            )
            clusters[cluster_index].append(pixel)

        relaxed: list[Pixel] = []
        for old_point, cluster in zip(selected, clusters):
            if not cluster:
                relaxed.append(old_point)
                continue

            center_x = sum(x for x, _ in cluster) / len(cluster)
            center_y = sum(y for _, y in cluster) / len(cluster)
            relaxed.append(
                min(
                    cluster,
                    key=lambda pixel: (pixel[0] - center_x) ** 2 + (pixel[1] - center_y) ** 2,
                ),
            )

        if relaxed == selected:
            break
        selected = relaxed

    selected = order_points_for_display(selected)
    return [(float(x), float(y)) for x, y in selected]


def squared_distance(a: Pixel, b: Pixel) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def squared_float_distance(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def topmost_pixel(pixels: set[Pixel]) -> Pixel:
    return min(pixels, key=lambda pixel: (pixel[1], pixel[0]))


def order_stemmed_letter_labels(points: list[Point], letter: str) -> list[Point]:
    """
    Put labels on stemmed branched letters in a connected visual order.

    The point placement is already even; this only reorders labels. For P/p/R/r
    the labels should behave like one walk: bottom stem, top stem,
    bowl/shoulder, then R's leg.
    """
    if len(points) < 4:
        return points

    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return points

    stem_x_limit = min_x + width * 0.24
    top_band_limit = min_y + height * 0.12
    if letter.lower() in {"p", "r"}:
        stem_points = [point for point in points if point[0] <= stem_x_limit]
    else:
        stem_points = [
            point
            for point in points
            if point[0] <= stem_x_limit and point[1] > top_band_limit
        ]

    if len(stem_points) < 2:
        return points

    stem_lookup = set(stem_points)
    non_stem_points = [point for point in points if point not in stem_lookup]
    ordered_stem = sorted(stem_points, key=lambda point: (-point[1], point[0]))

    if not non_stem_points:
        return ordered_stem

    if letter == "R":
        leg_x_limit = min_x + width * 0.48
        leg_y_limit = min_y + height * 0.58
        leg_points = [
            point
            for point in non_stem_points
            if point[0] > leg_x_limit and point[1] > leg_y_limit
        ]
        leg_lookup = set(leg_points)
        shoulder_points = [point for point in non_stem_points if point not in leg_lookup]
        ordered = order_points_around_center_from_reference(shoulder_points, ordered_stem[-1])
        ordered.extend(sorted(leg_points, key=lambda point: (point[1], point[0])))
    else:
        ordered = order_points_around_center_from_reference(non_stem_points, ordered_stem[-1])

    return ordered_stem + ordered


def order_points_around_center_from_upper_left(points: list[Point]) -> list[Point]:
    """Order points around their center, starting from the upper-left point."""
    if len(points) < 3:
        return points

    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    ordered = sorted(
        points,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )
    start_index = min(
        range(len(ordered)),
        key=lambda index: (ordered[index][0] + ordered[index][1], ordered[index][1], ordered[index][0]),
    )
    return ordered[start_index:] + ordered[:start_index]


def order_points_around_center_from_reference(points: list[Point], reference: Point) -> list[Point]:
    """Order points around their center, starting nearest to a reference point."""
    if len(points) < 3:
        return sorted(points, key=lambda point: squared_float_distance(point, reference))

    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    ordered = sorted(
        points,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )
    start_index = min(
        range(len(ordered)),
        key=lambda index: squared_float_distance(ordered[index], reference),
    )
    return ordered[start_index:] + ordered[:start_index]


def order_points_for_display(points: list[Pixel]) -> list[Pixel]:
    """
    Give graph-sampled points a stable visual order without pretending that
    looped letters have one natural stroke path.
    """
    if len(points) < 3:
        return points

    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    ordered = sorted(
        points,
        key=lambda pixel: math.atan2(pixel[1] - center_y, pixel[0] - center_x),
    )

    start_index = min(range(len(ordered)), key=lambda index: (ordered[index][1], ordered[index][0]))
    return ordered[start_index:] + ordered[:start_index]


def skeleton_pixels(skeleton: list[bytearray]) -> set[Pixel]:
    pixels: set[Pixel] = set()

    for y, row in enumerate(skeleton):
        for x, value in enumerate(row):
            if value:
                pixels.add((x, y))

    return pixels


def largest_component(pixels: set[Pixel]) -> set[Pixel]:
    remaining = set(pixels)
    largest: set[Pixel] = set()

    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]

        while stack:
            pixel = stack.pop()
            for neighbor in graph_neighbors(pixel, remaining):
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)

        if len(component) > len(largest):
            largest = component

    return largest


def pixel_degree(pixel: Pixel, pixels: set[Pixel]) -> int:
    return sum(1 for _ in graph_neighbors(pixel, pixels))


def graph_neighbors(pixel: Pixel, pixels: set[Pixel]) -> Iterable[Pixel]:
    x, y = pixel

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            neighbor = (x + dx, y + dy)
            if neighbor in pixels:
                yield neighbor


def dijkstra_skeleton(pixels: set[Pixel], start: Pixel) -> tuple[dict[Pixel, float], dict[Pixel, Pixel]]:
    distances: dict[Pixel, float] = {start: 0.0}
    previous: dict[Pixel, Pixel] = {}
    heap: list[tuple[float, Pixel]] = [(0.0, start)]

    while heap:
        current_distance, current = heapq.heappop(heap)
        if current_distance != distances[current]:
            continue

        cx, cy = current
        for neighbor in graph_neighbors(current, pixels):
            nx, ny = neighbor
            step = math.hypot(nx - cx, ny - cy)
            next_distance = current_distance + step

            if next_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = next_distance
                previous[neighbor] = current
                heapq.heappush(heap, (next_distance, neighbor))

    return distances, previous


def rebuild_path(previous: dict[Pixel, Pixel], start: Pixel, end: Pixel) -> list[Pixel]:
    path = [end]
    current = end

    while current != start:
        current = previous.get(current)
        if current is None:
            return []
        path.append(current)

    path.reverse()
    return path


def orient_top_to_bottom(path: list[Pixel]) -> list[Pixel]:
    if len(path) < 2:
        return path

    first = path[0]
    last = path[-1]

    if first[1] > last[1]:
        return list(reversed(path))
    if first[1] == last[1] and first[0] > last[0]:
        return list(reversed(path))
    return path


def smooth_polyline(points: list[Point], *, iterations: int = 2) -> list[Point]:
    """Lightly smooth the pixel skeleton while keeping endpoints fixed."""
    output = points

    for _ in range(max(0, iterations)):
        if len(output) < 3:
            return output

        smoothed: list[Point] = [output[0]]
        for a, b in zip(output, output[1:]):
            smoothed.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            smoothed.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        smoothed.append(output[-1])
        output = smoothed

    return output


def sample_polyline(
    points: list[Point],
    number_of_points: int,
    *,
    include_endpoints: bool = True,
) -> list[Point]:
    """Sample a polyline by equal arc-length spacing."""
    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive.")
    if len(points) < 2:
        raise ValueError("Need at least two source points to sample a polyline.")

    lengths = [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    ]
    total_length = sum(lengths)

    if total_length <= 0:
        raise ValueError("Polyline has no measurable length.")

    sampled: list[Point] = []
    segment_index = 0
    segment_start_distance = 0.0

    for i in range(number_of_points):
        if include_endpoints and number_of_points > 1:
            target = total_length * i / (number_of_points - 1)
        else:
            target = total_length * i / number_of_points

        while (
            segment_index < len(lengths) - 1
            and segment_start_distance + lengths[segment_index] < target
        ):
            segment_start_distance += lengths[segment_index]
            segment_index += 1

        a = points[segment_index]
        b = points[segment_index + 1]
        length = lengths[segment_index]
        t = 0.0 if length == 0 else (target - segment_start_distance) / length
        sampled.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

    return sampled


def draw_points_overlay(
    image: Image.Image,
    points: list[Point],
    *,
    connect_points: bool = False,
    point_radius: int = 5,
    point_fill: tuple[int, int, int, int] = (0, 96, 220, 255),
    line_fill: tuple[int, int, int, int] = (240, 90, 40, 120),
) -> Image.Image:
    """Optional helper: draw the returned points over the returned font image."""
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    if connect_points and len(points) >= 2:
        draw.line(points, fill=line_fill, width=max(1, point_radius // 2))

    for x, y in points:
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=point_fill,
            outline=(255, 255, 255, 255),
            width=2,
        )

    return overlay


def ask_positive_int(prompt: str) -> int:
    """Prompt until the user enters a positive integer."""
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("Please enter a whole number, for example 12.")
            continue

        if number > 0:
            return number

        print("Please enter a number greater than 0.")


def run_interactive_export() -> None:
    """Ask for the point count, then generate the A-Z DataFrame and images."""
    default_font = Path(r"C:\Windows\Fonts\arial.ttf")

    number_of_points = ask_positive_int("Number of points per letter: ")
    font_input = input(f"Font path [{default_font}]: ").strip()
    font = Path(font_input) if font_input else default_font

    if not font.exists():
        raise FileNotFoundError(f"Font file was not found: {font}")

    output_dir = Path(f"letter_images_{number_of_points}_points")
    csv_path = Path(f"letter_dataframe_{number_of_points}_points.csv")

    df = letter_points_dataframe(
        number_of_points,
        font,
        output_dir=output_dir,
        coordinate_system="cartesian",
        fixed_canvas=True,
    )
    df.to_csv(csv_path, index=False)

    print()
    print(f"Generated {len(df)} rows.")
    print(f"Saved DataFrame CSV: {csv_path.resolve()}")
    print(f"Saved letter images: {output_dir.resolve()}")


if __name__ == "__main__":
    run_interactive_export()
