# Letter Point Generator

Generate `(x, y)` coordinates and matching images for points placed along real font-rendered letter shapes.

This project helps when you need circles, markers, or other visual elements to follow the shape of letters in a dashboard, visualization tool, report, or presentation. The script renders letters from a font, finds a usable centerline through each glyph, samples evenly distributed points, and exports both the coordinate table and aligned letter images.

## What It Creates

For a requested point count, the script exports:

```text
letter_dataframe_<number>_points.csv
letter_images_<number>_points/
```

The CSV uses this simple column format:

```text
number_of_points, label, letter, x_axis, y_axis
```

The image folder contains one image per letter. The images and coordinates share the same coordinate frame, so they can be imported into visualization tools together.

## Install

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python letter_points.py
```

The script asks for:

```text
Number of points per letter:
Font path [C:\Windows\Fonts\arial.ttf]:
```

Press Enter at the font prompt to use Arial on Windows.

## Example

If you enter `12`, the script generates:

```text
letter_dataframe_12_points.csv
letter_images_12_points/
```

## Why It Helps

Many visualization tools can import coordinates and images, but they do not have an easy way to generate letter-shaped marker layouts. This project creates those assets ahead of time so the layout is repeatable and easy to reuse.

## Project History

The first Git commit contains the earlier hand-authored template/scatter implementation. The current commit replaces it with this font-based generator. Use Git history to switch between them:

```powershell
git log --oneline
git checkout <commit-id>
```
