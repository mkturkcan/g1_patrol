"""Procedural New York street litter: coffee cups, soda cans, water bottles, a bottle in a paper bag, takeout
containers, deli bags, crumpled napkins and chip bags; and tied trash bags (black, and a blue recycling bag).

Writes one USD file per item (``<out>/<name>.usda``): textured visual meshes, a convex collision hull and the mass,
plus the textures (``<out>/textures``) and ``manifest.json`` (size, mass and the height of the item's center when
lying on the ground). All geometry and textures are generated here; the designs are generic (no brands).

Usage:
    python scripts/assets/make_nyc_trash.py --out src/g1_patrol/assets/data/trash
"""

import argparse
import json
import math
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEX = 512


# ---------------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------------


def lathe(profile, segments=32, cap_bottom=True, cap_top=True):
    """Surface of revolution about z from (radius, z) points listed bottom to top. UVs: u around, v along the profile."""
    profile = np.asarray(profile, dtype=float)
    k = len(profile)
    angles = np.linspace(0.0, 2 * np.pi, segments + 1)  # seam column duplicated for the texture
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))])
    v = arc / arc[-1]
    verts, uvs = [], []
    for i, (r, z) in enumerate(profile):
        for j, a in enumerate(angles):
            verts.append((r * math.cos(a), r * math.sin(a), z))
            uvs.append((j / segments, v[i]))
    faces = []
    w = segments + 1
    for i in range(k - 1):
        for j in range(segments):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            faces += [(a, b, c), (a, c, d)]
    verts, uvs = list(verts), list(uvs)
    for cap, i, flip in ((cap_bottom, 0, True), (cap_top, k - 1, False)):
        if not cap or profile[i, 0] < 1e-6:
            continue
        center = len(verts)
        verts.append((0.0, 0.0, profile[i, 1]))
        uvs.append((0.5, 0.0 if flip else 1.0))
        for j in range(segments):
            a, b = i * w + j, i * w + j + 1
            faces.append((center, b, a) if flip else (center, a, b))
    return np.array(verts), np.array(faces), np.array(uvs)


def sphere(radius, subdivisions=3):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    v = np.array(m.vertices)
    uv = np.stack([np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi) + 0.5, np.arccos(np.clip(v[:, 2] / radius, -1, 1)) / np.pi], 1)
    return v, np.array(m.faces), uv


def noise3(points, seed, octaves=3, scale=12.0):
    """Smooth pseudo-random scalar field on 3D points (sum of random sinusoids)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(points))
    for o in range(octaves):
        for _ in range(6):
            k = rng.normal(size=3) * scale * (2**o)
            out += np.sin(points @ k + rng.uniform(0, 2 * np.pi)) / (2**o) / 6
    return out


def displace(verts, amount, seed, scale=12.0):
    n = verts / np.linalg.norm(verts, axis=1, keepdims=True).clip(1e-9)
    return verts + n * (amount * noise3(verts, seed, scale=scale))[:, None]


def frustum_box(bottom, top, height, segments_v=4):
    """Open-top tapered box (takeout pail): rectangular rings from bottom (bx, by) to top (tx, ty)."""
    rings = []
    for t in np.linspace(0, 1, segments_v + 1):
        x = bottom[0] + (top[0] - bottom[0]) * t
        y = bottom[1] + (top[1] - bottom[1]) * t
        rings.append([(-x, -y, t * height), (x, -y, t * height), (x, y, t * height), (-x, y, t * height), (-x, -y, t * height)])
    verts, uvs, faces = [], [], []
    w = 5
    perim = np.array([0, 1, 2, 3, 4]) / 4
    for i, ring in enumerate(rings):
        for j, p in enumerate(ring):
            verts.append(p)
            uvs.append((perim[j], i / segments_v))
    for i in range(segments_v):
        for j in range(4):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            faces += [(a, b, c), (a, c, d)]
    c0 = len(verts)
    verts += [(-bottom[0], -bottom[1], 0), (bottom[0], -bottom[1], 0), (bottom[0], bottom[1], 0), (-bottom[0], bottom[1], 0)]
    uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
    faces += [(c0, c0 + 2, c0 + 1), (c0, c0 + 3, c0 + 2)]
    return np.array(verts, dtype=float), np.array(faces), np.array(uvs, dtype=float)


def pillow(lx, ly, lz, seed, nu=24, nv=16, crimp=0.12):
    """Snack bag: a flat pillow with crimped seals at both ends and crinkles."""
    verts, uvs, faces = [], [], []
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            x = (u - 0.5) * lx
            y = (v - 0.5) * ly
            seal = min(u, 1 - u) < crimp
            puff = 0.0 if seal else math.sin(math.pi * (u - crimp) / (1 - 2 * crimp)) ** 0.6 * math.sin(math.pi * v) ** 0.6
            verts.append((x, y, lz / 2 * puff))
            uvs.append((u, v * 0.5))
    top = len(verts)
    for i in range(nu + 1):
        for j in range(nv + 1):
            x, y, z = verts[i * (nv + 1) + j]
            verts.append((x, y, -z))
            uvs.append((i / nu, 0.5 + j / nv * 0.5))
    w = nv + 1
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            faces += [(a, b, c), (a, c, d)]
            a, b, c, d = top + a, top + b, top + c, top + d
            faces += [(a, c, b), (a, d, c)]
    verts = np.array(verts, dtype=float)
    verts[:, 2] += 0.004 * noise3(verts, seed, scale=60.0)
    return verts, np.array(faces), np.array(uvs, dtype=float)


# ---------------------------------------------------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------------------------------------------------


def greek_key_cup(path):
    """Blue paper coffee cup with a white Greek-key (meander) band."""
    img = Image.new("RGB", (TEX, TEX), (28, 74, 158))
    d = ImageDraw.Draw(img)
    band_top, band_bottom = int(TEX * 0.38), int(TEX * 0.62)
    d.rectangle([0, band_top - 10, TEX, band_top - 4], fill=(245, 245, 240))
    d.rectangle([0, band_bottom + 4, TEX, band_bottom + 10], fill=(245, 245, 240))
    cell = (band_bottom - band_top) // 2
    x = 0
    while x < TEX:  # meander: a square spiral hook per cell
        h = band_bottom - band_top
        pts = [(x, band_bottom), (x, band_top), (x + h, band_top), (x + h, band_bottom - cell), (x + cell, band_bottom - cell),
               (x + cell, band_top + cell // 2), (x + h - cell // 2, band_top + cell // 2)]
        d.line(pts, fill=(245, 245, 240), width=6)
        d.line([(x, band_bottom), (x + h + cell // 2, band_bottom)], fill=(245, 245, 240), width=6)
        x += h + cell // 2
    d.rectangle([0, TEX - 18, TEX, TEX], fill=(240, 240, 235))  # rolled rim
    img.save(path)


def can_texture(path, base, accent, seed):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (TEX, TEX), base)
    d = ImageDraw.Draw(img)
    phase = rng.uniform(0, 2 * np.pi)
    pts = [(x, int(TEX * (0.5 + 0.12 * math.sin(phase + 4 * math.pi * x / TEX)))) for x in range(0, TEX + 8, 8)]
    d.line(pts, fill=accent, width=34)
    d.rectangle([0, 0, TEX, 26], fill=(190, 192, 196))
    d.rectangle([0, TEX - 26, TEX, TEX], fill=(190, 192, 196))
    img.save(path)


def label_texture(path, base, band, text, text_color):
    img = Image.new("RGBA", (TEX, TEX), base)
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(TEX * 0.35), TEX, int(TEX * 0.62)], fill=band)
    font = ImageFont.truetype(FONT, 34)
    for k in range(2):
        d.text((20 + k * TEX // 2, int(TEX * 0.44)), text, fill=text_color, font=font)
    img.save(path)


def kraft_texture(path, seed, base=(160, 120, 78)):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (TEX // 8, TEX // 8))
    img = Image.fromarray(((a - a.min()) / (np.ptp(a) + 1e-9) * 255).astype(np.uint8)).resize((TEX, TEX), Image.BICUBIC)
    n = np.asarray(img.filter(ImageFilter.GaussianBlur(2)), dtype=float) / 255 - 0.5
    rgb = np.clip(np.array(base)[None, None] * (1 + 0.25 * n[..., None]), 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def thank_you_bag(path):
    img = Image.new("RGBA", (TEX, TEX), (244, 244, 240, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 40)
    for row, y in enumerate((150, 210, 270)):
        d.text((60 + (row % 2) * 30, y), "THANK YOU", fill=(200, 30, 36), font=font)
    img.save(path)


def chip_texture(path, base, accent):
    img = Image.new("RGB", (TEX, TEX), base)
    d = ImageDraw.Draw(img)
    for k in range(6):
        y = 60 + 70 * k
        d.ellipse([TEX * 0.25, y, TEX * 0.75, y + 50], outline=accent, width=8)
    d.rectangle([0, 0, int(TEX * 0.12), TEX], fill=(210, 210, 215))
    d.rectangle([int(TEX * 0.88), 0, TEX, TEX], fill=(210, 210, 215))
    img.save(path)


def plain(path, color):
    Image.new("RGB", (16, 16), color).save(path)


# ---------------------------------------------------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------------------------------------------------


def coffee_cup():
    r0, r1, h = 0.030, 0.045, 0.115
    body = lathe([(r0, 0.0), (r1, h)], 32, cap_bottom=True, cap_top=False)
    lid = lathe([(r1 + 0.002, h), (r1 + 0.003, h + 0.006), (r1 - 0.006, h + 0.012), (0.012, h + 0.014), (0.0, h + 0.014)], 32,
                cap_bottom=False, cap_top=False)
    return [("body", body, "cup_body"), ("lid", lid, "white_plastic")], 0.015


def soda_can(crushed=False, seed=0):
    r, h = 0.033, 0.122
    prof = [(0.0, 0.0), (0.026, 0.0), (0.033, 0.010), (r, 0.110), (0.028, h), (0.0, h)]
    v, f, uv = lathe(prof, 32, cap_bottom=False, cap_top=False)
    if crushed:
        v[:, 2] *= 0.55
        v[:, :2] *= (1.0 + 0.25 * noise3(v, seed, scale=40.0))[:, None]
    return [("body", (v, f, uv), "can")], 0.015


def water_bottle():
    prof = [(0.0, 0.0), (0.030, 0.0), (0.0325, 0.006), (0.0325, 0.150), (0.026, 0.175), (0.013, 0.192), (0.013, 0.200)]
    body = lathe(prof, 32, cap_bottom=False, cap_top=False)
    cap = lathe([(0.0135, 0.200), (0.0140, 0.212), (0.0, 0.212)], 24, cap_bottom=False, cap_top=False)
    return [("body", body, "bottle"), ("cap", cap, "blue_plastic")], 0.02


def bottle_in_bag(seed=3):
    prof = [(0.0, 0.0), (0.042, 0.0), (0.047, 0.01), (0.044, 0.17), (0.034, 0.20), (0.026, 0.215), (0.0, 0.215)]
    v, f, uv = lathe(prof, 32, cap_bottom=False, cap_top=False)
    v[:, :2] *= (1.0 + 0.06 * noise3(v, seed, scale=30.0))[:, None]
    neck = lathe([(0.012, 0.205), (0.012, 0.255), (0.013, 0.258), (0.0, 0.258)], 20, cap_bottom=False, cap_top=False)
    return [("bag", (v, f, uv), "kraft"), ("neck", neck, "green_glass")], 0.25


def takeout_pail():
    body = frustum_box((0.035, 0.030), (0.052, 0.045), 0.10)
    handle = trimesh.creation.torus(major_radius=0.04, minor_radius=0.0015, major_sections=24, minor_sections=6)
    hv = np.array(handle.vertices)
    hv = hv[:, [0, 2, 1]]  # stand the loop up
    hv[:, 2] += 0.10
    keep = hv[:, 2] >= 0.10
    hf = np.array([face for face in handle.faces if keep[face].all()])
    huv = np.zeros((len(hv), 2))
    return [("body", body, "pail"), ("handle", (hv, hf, huv), "wire")], 0.012


def deli_bag(seed=5):
    """Knotted plastic bag: a lumpy body, a small knot and the two handle tails sticking out of it."""
    v, f, uv = sphere(0.07, 3)
    v[:, 2] *= 0.75
    v[:, 0] *= 1.25
    v = displace(v, 0.012, seed, scale=25.0)
    parts = [("bag", (v, f, uv), "deli_bag")]
    kv, kf, kuv = sphere(0.013, 2)
    kv = displace(kv, 0.003, seed + 1, scale=80.0) + (0.0, 0.0, 0.056)
    parts.append(("knot", (kv, kf, kuv), "deli_bag"))
    for side in (-1, 1):  # handle tails: thin crumpled flaps leaning out of the knot
        tv, tf, tuv = sphere(0.02, 2)
        tv = displace(tv, 0.004, seed + 2 + side, scale=70.0)
        tv *= (0.55, 0.18, 1.3)
        c, s_ = math.cos(0.6 * side), math.sin(0.6 * side)
        tv = tv @ np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
        tv += (0.018 * side, 0.0, 0.075)
        parts.append(("tail%d" % (side + 1), (tv, tf, tuv), "deli_bag"))
    return parts, 0.03


def crumpled_napkin(seed=7):
    v, f, uv = sphere(0.032, 3)
    v = displace(v, 0.009, seed, scale=50.0)
    return [("paper", (v, f, uv), "napkin")], 0.008


def chip_bag(seed=11):
    return [("bag", pillow(0.16, 0.11, 0.035, seed), "chip_bag")], 0.012


def foam_clamshell():
    m = trimesh.creation.box(extents=(0.23, 0.15, 0.065))
    v = np.array(m.vertices)
    v[:, 2] += 0.0325
    v[:, :2] *= np.where(v[:, 2:3] > 0.03, 0.95, 1.0)
    uv = np.stack([v[:, 0] / 0.23 + 0.5, v[:, 1] / 0.15 + 0.5], 1)
    return [("box", (v, np.array(m.faces), uv), "foam")], 0.01


def trash_bag(width, depth, height, seed, material, mass):
    """A full, tied trash bag standing on the ground: a lumpy body that sags and spreads at the bottom, gathered into
    a neck at the top, a knot and the two tie ends sticking out of it."""
    profile = [(0.0, 0.0), (0.40, 0.004), (0.49, 0.05), (0.52, 0.22), (0.50, 0.45), (0.42, 0.64), (0.28, 0.78),
               (0.12, 0.87), (0.05, 0.91), (0.03, 0.93)]
    profile = [(r * width, z * height) for r, z in profile]
    v, f, uv = lathe(profile, segments=48, cap_top=False)
    v[:, 1] *= depth / width
    # lumps of the contents and wrinkles of the film, outwards from the bag's axis (none at the bottom center)
    radial = np.zeros_like(v)
    radial[:, :2] = v[:, :2] / np.linalg.norm(v[:, :2], axis=1, keepdims=True).clip(1e-6)
    lumps = 0.045 * width * noise3(v / height, seed, octaves=2, scale=5.0)
    wrinkles = 0.006 * noise3(v / height, seed + 1, octaves=2, scale=40.0)
    bottom = np.clip(v[:, 2] / (0.05 * height), 0.0, 1.0)
    v += radial * ((lumps + wrinkles) * bottom)[:, None]
    parts = [("bag", (v, f, uv), material)]
    top = 0.93 * height
    kv, kf, kuv = sphere(0.022, 2)
    kv = displace(kv, 0.004, seed + 2, scale=70.0) + (0.0, 0.0, top + 0.012)
    parts.append(("knot", (kv, kf, kuv), material))
    for side in (-1, 1):  # tie ends: thin crumpled flaps leaning out of the knot
        tv, tf, tuv = sphere(0.03, 2)
        tv = displace(tv, 0.006, seed + 3 + side, scale=60.0)
        tv *= (0.7, 0.15, 1.2)
        c, s_ = math.cos(0.7 * side), math.sin(0.7 * side)
        tv = tv @ np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
        tv += (0.03 * side, 0.0, top + 0.035)
        parts.append(("tie%d" % (side + 1), (tv, tf, tuv), material))
    return parts, mass


BAGS = {
    "trash_bag_small": lambda: trash_bag(0.30, 0.27, 0.40, 21, "black_bag", 1.4),
    "trash_bag_medium": lambda: trash_bag(0.36, 0.32, 0.46, 23, "black_bag", 2.2),
    "recycling_bag": lambda: trash_bag(0.34, 0.31, 0.44, 25, "blue_bag", 1.0),
}
"""Trash bags (manifest ``kind`` "bag"): set out at the curb, carried by the knot."""

ITEMS = {
    "coffee_cup": coffee_cup,
    "soda_can_red": lambda: soda_can(False, 1),
    "soda_can_crushed": lambda: soda_can(True, 2),
    "water_bottle": water_bottle,
    "bottle_in_bag": bottle_in_bag,
    "takeout_pail": takeout_pail,
    "deli_bag": deli_bag,
    "crumpled_napkin": crumpled_napkin,
    "chip_bag": chip_bag,
    "foam_clamshell": foam_clamshell,
}

MATERIALS = {  # name -> (texture file or None, color, roughness, metallic, opacity)
    "cup_body": ("coffee_cup.png", None, 0.55, 0.0, 1.0),
    "white_plastic": (None, (0.93, 0.93, 0.91), 0.4, 0.0, 1.0),
    "can": ("can_red.png", None, 0.3, 0.6, 1.0),
    "bottle": ("water_bottle.png", None, 0.1, 0.0, 0.55),
    "blue_plastic": (None, (0.12, 0.30, 0.75), 0.4, 0.0, 1.0),
    "kraft": ("kraft.png", None, 0.85, 0.0, 1.0),
    "green_glass": (None, (0.10, 0.30, 0.12), 0.1, 0.0, 0.8),
    "pail": ("takeout_pail.png", None, 0.6, 0.0, 1.0),
    "wire": (None, (0.6, 0.6, 0.62), 0.3, 0.9, 1.0),
    "deli_bag": ("deli_bag.png", None, 0.45, 0.0, 0.9),
    "napkin": ("napkin.png", None, 0.9, 0.0, 1.0),
    "chip_bag": ("chip_bag.png", None, 0.25, 0.5, 1.0),
    "foam": (None, (0.95, 0.95, 0.93), 0.7, 0.0, 1.0),
    "black_bag": (None, (0.018, 0.018, 0.021), 0.28, 0.0, 1.0),
    "blue_bag": (None, (0.16, 0.34, 0.68), 0.3, 0.0, 0.85),
}


def write_textures(tex_dir):
    os.makedirs(tex_dir, exist_ok=True)
    greek_key_cup(os.path.join(tex_dir, "coffee_cup.png"))
    can_texture(os.path.join(tex_dir, "can_red.png"), (186, 24, 32), (245, 245, 245), 1)
    label_texture(os.path.join(tex_dir, "water_bottle.png"), (200, 225, 240, 255), (30, 110, 200, 255), "SPRING WATER", (255, 255, 255, 255))
    kraft_texture(os.path.join(tex_dir, "kraft.png"), 3)
    kraft_texture(os.path.join(tex_dir, "napkin.png"), 7, base=(232, 226, 214))
    label_texture(os.path.join(tex_dir, "takeout_pail.png"), (246, 246, 242, 255), (246, 246, 242, 255), "", (0, 0, 0, 255))
    img = Image.open(os.path.join(tex_dir, "takeout_pail.png"))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(TEX * 0.70), TEX, int(TEX * 0.72)], fill=(196, 30, 40, 255))
    d.rectangle([0, int(TEX * 0.76), TEX, int(TEX * 0.78)], fill=(196, 30, 40, 255))
    img.save(os.path.join(tex_dir, "takeout_pail.png"))
    thank_you_bag(os.path.join(tex_dir, "deli_bag.png"))
    chip_texture(os.path.join(tex_dir, "chip_bag.png"), (236, 168, 28), (200, 60, 30))


# ---------------------------------------------------------------------------------------------------------------------
# USD (ASCII) writer
# ---------------------------------------------------------------------------------------------------------------------


def _vec(arr, fmt="%.5f"):
    return "[" + ", ".join("(" + ", ".join(fmt % x for x in row) + ")" for row in arr) + "]"


def mesh_usda(name, verts, faces, uvs, material, indent="    "):
    m = trimesh.Trimesh(verts, faces, process=False)
    normals = np.array(m.vertex_normals)
    return f"""{indent}def Mesh "{name}"
{indent}{{
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in np.asarray(faces).reshape(-1))}]
{indent}    point3f[] points = {_vec(verts)}
{indent}    normal3f[] normals = {_vec(normals)} (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    texCoord2f[] primvars:st = {_vec(uvs)} (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}    rel material:binding = </trash/Looks/{material}>
{indent}}}
"""


def material_usda(name, spec, indent="        "):
    tex, color, rough, metal, opacity = spec
    root = f"/trash/Looks/{name}"
    if tex:
        diffuse = f"color3f inputs:diffuseColor.connect = <{root}/tex.outputs:rgb>"
        shaders = f"""
{indent}def Shader "tex"
{indent}{{
{indent}    uniform token info:id = "UsdUVTexture"
{indent}    asset inputs:file = @./textures/{tex}@
{indent}    float2 inputs:st.connect = <{root}/st.outputs:result>
{indent}    token inputs:wrapS = "repeat"
{indent}    token inputs:wrapT = "clamp"
{indent}    float3 outputs:rgb
{indent}}}
{indent}def Shader "st"
{indent}{{
{indent}    uniform token info:id = "UsdPrimvarReader_float2"
{indent}    string inputs:varname = "st"
{indent}    float2 outputs:result
{indent}}}"""
    else:
        diffuse = "color3f inputs:diffuseColor = (%.3f, %.3f, %.3f)" % color
        shaders = ""
    return f"""    def Material "{name}"
    {{
        token outputs:surface.connect = <{root}/shader.outputs:surface>
{indent}def Shader "shader"
{indent}{{
{indent}    uniform token info:id = "UsdPreviewSurface"
{indent}    {diffuse}
{indent}    float inputs:roughness = {rough}
{indent}    float inputs:metallic = {metal}
{indent}    float inputs:opacity = {opacity}
{indent}    token outputs:surface
{indent}}}{shaders}
    }}
"""


def write_item(out_dir, name, parts, mass):
    # center the item: origin at the center of its bounding box
    allv = np.concatenate([p[1][0] for p in parts])
    center = (allv.min(0) + allv.max(0)) / 2
    parts = [(pn, (v - center, f, uv), mat) for pn, (v, f, uv), mat in parts]
    allv = np.concatenate([p[1][0] for p in parts])
    hull = trimesh.convex.convex_hull(allv)
    size = allv.max(0) - allv.min(0)
    meshes = "".join(mesh_usda(pn, v, f, uv, mat, indent="    ") for pn, (v, f, uv), mat in parts)
    mats = "".join(material_usda(m, MATERIALS[m]) for m in sorted({p[2] for p in parts}))
    usda = f"""#usda 1.0
(
    defaultPrim = "trash"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "trash" (
    kind = "component"
    prepend apiSchemas = ["PhysicsMassAPI"]
)
{{
    float physics:mass = {mass}
{meshes}    def Mesh "collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexHull"
        uniform token purpose = "guide"
        int[] faceVertexCounts = [{", ".join("3" for _ in hull.faces)}]
        int[] faceVertexIndices = [{", ".join(str(int(i)) for i in np.asarray(hull.faces).reshape(-1))}]
        point3f[] points = {_vec(hull.vertices)}
    }}
    def Scope "Looks"
    {{
{mats}    }}
}}
"""
    with open(os.path.join(out_dir, f"{name}.usda"), "w") as fh:
        fh.write(usda)
    # resting heights of the center on its three sides (smallest dimension down is the usual resting pose)
    return {"file": f"{name}.usda", "size": [round(float(s), 4) for s in size], "mass": mass,
            "rest_height": round(float(min(size)) / 2, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "src", "g1_patrol", "assets", "data", "trash"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_textures(os.path.join(args.out, "textures"))
    manifest = {}
    for name, make in ITEMS.items():
        parts, mass = make()
        manifest[name] = write_item(args.out, name, parts, mass)
        print(name, manifest[name])
    for name, make in BAGS.items():
        parts, mass = make()
        manifest[name] = write_item(args.out, name, parts, mass) | {"kind": "bag"}
        print(name, manifest[name])
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
