"""Behavioral regression checks for periodic motif composition, without ImageGen."""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parent / "seamless-patterns" / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("compose_pattern", SCRIPTS / "compose_pattern.py")
composer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composer)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="composition-tests-", dir=ROOT)
        self.work = Path(self.temp.name)
        image = Image.new("RGBA", (8, 10), (87, 51, 199, 0))
        for y in range(1, 9):
            for x in range(1, 7):
                image.putpixel((x, y), (x * 31, y * 23, 109, 67 + (x + y) % 4 * 53))
        image.save(self.work / "motif.png")
        self.config = {
            "schema_version": 1, "width": 37, "height": 29,
            "background": "#FFFFFF", "seed": 42,
            "assets": [{"id": "motif", "path": "motif.png"}],
            "layout": {"count": 7, "size_px": [6, 15], "rotation_deg": [-25, 35],
                       "min_center_distance_px": 5, "max_attempts": 1000},
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_config(self, name, config=None):
        path = self.work / (name + ".json")
        path.write_text(json.dumps(config or self.config), encoding="utf-8")
        manifest = composer.compose_pattern(path, self.work / name)
        return json.loads(manifest.read_text(encoding="utf-8")), manifest.parent

    def test_same_seed_and_explicit_replay_reproduce_pixels(self):
        original = (self.work / "motif.png").read_bytes()
        first, first_dir = self.run_config("first")
        second, second_dir = self.run_config("second")
        self.assertEqual(first["placements"], second["placements"])
        self.assertEqual((first_dir / "tile.png").read_bytes(), (second_dir / "tile.png").read_bytes())
        replay_path = composer.compose_pattern(first_dir / "replay-config.json", self.work / "replay")
        self.assertEqual((first_dir / "tile.png").read_bytes(), (replay_path.parent / "tile.png").read_bytes())
        changed = json.loads(json.dumps(self.config))
        changed["seed"] = 43
        third, _ = self.run_config("different", changed)
        self.assertNotEqual(first["placements"], third["placements"])
        self.assertEqual(original, (self.work / "motif.png").read_bytes())
        self.assertEqual(first["visual_review_status"], "pending")
        report = json.loads((first_dir / "check" / "inspection.json").read_text(encoding="utf-8"))
        self.assertTrue(report["assembly"]["pixel_exact_copies_verified"])
        self.assertEqual(report["visual_review"]["repeat_distribution"], "pending")
        self.assertEqual(report["diagnostics"]["background_distribution"]["verdict"], "not_assessed")

    def test_corner_crossing_rotation_large_motifs_and_layer_order_match_infinite_field(self):
        # Independent reference: draw many translated motifs on a larger canvas,
        # then compare every tile to the single-period output (including corners).
        assets = composer.load_assets(self.config["assets"], self.work)
        for width, height, size, angle in ((17, 13, 8, 0), (17, 13, 11, 37), (7, 5, 23, 0)):
            with self.subTest(period=(width, height), size=size, angle=angle):
                placements = [
                    {"asset_id": "motif", "x": 0, "y": 0, "size_px": size, "rotation_deg": angle},
                    {"asset_id": "motif", "x": width - 1, "y": height - 1,
                     "size_px": size + 2, "rotation_deg": -angle},
                ]
                actual, resolved = composer.render_placements(assets, placements, width, height, (0, 0, 0, 0))
                reference = Image.new("RGBA", (3 * width, 3 * height))
                for item in placements:
                    source = assets["motif"]["image"]
                    factor = item["size_px"] / max(source.size)
                    shape = source.resize((max(1, round(source.width * factor)), max(1, round(source.height * factor))),
                                          Image.Resampling.LANCZOS)
                    if item["rotation_deg"] % 360:
                        shape = shape.rotate(item["rotation_deg"], Image.Resampling.BICUBIC, expand=True)
                    periods = math.ceil(max(shape.size) / min(width, height)) + 4
                    for row in range(-periods, periods + 1):
                        for column in range(-periods, periods + 1):
                            left = item["x"] - shape.width // 2 + column * width
                            top = item["y"] - shape.height // 2 + row * height
                            if left < reference.width and top < reference.height and left + shape.width > 0 and top + shape.height > 0:
                                reference.alpha_composite(shape, dest=(left, top))
                for row in range(3):
                    for column in range(3):
                        block = reference.crop((column * width, row * height, (column + 1) * width, (row + 1) * height))
                        self.assertEqual(block.tobytes(), actual.tobytes())
                self.assertGreater(len(resolved[0]["origins_xy"]), 1)
                if size > max(width, height):
                    self.assertGreater(len(resolved[0]["origins_xy"]), 4)

    def test_toroidal_spacing_counts_neighbors_across_edges(self):
        assets = composer.load_assets(self.config["assets"], self.work)
        layout = {"count": 20, "size_px": [2, 3], "min_center_distance_px": 10, "max_attempts": 10000}
        items = composer.generate_placements(layout, 10, 60, 70, assets)
        for index, item in enumerate(items):
            for other in items[:index]:
                dx, dy = abs(item["x"] - other["x"]), abs(item["y"] - other["y"])
                self.assertGreaterEqual(min(dx, 60 - dx) ** 2 + min(dy, 70 - dy) ** 2, 100)
        impossible = {"count": 2, "size_px": [2, 2], "min_center_distance_px": 100, "max_attempts": 20}
        with self.assertRaisesRegex(ValueError, "Placed only 1 of 2"):
            composer.generate_placements(impossible, 1, 10, 10, assets)

    def test_changed_asset_is_rejected_on_replay(self):
        _, folder = self.run_config("first")
        Image.new("RGBA", (8, 10), (0, 0, 0, 0)).save(self.work / "motif.png")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            composer.compose_pattern(folder / "replay-config.json", self.work / "replay")
        self.assertFalse((self.work / "replay").exists())

    def test_flattened_or_clipped_motifs_and_unknown_config_are_rejected(self):
        for name, image in (("white", Image.new("RGB", (7, 7), "white")),
                            ("clipped", Image.new("RGBA", (7, 7), (240, 15, 20, 255))),
                            ("empty", Image.new("RGBA", (7, 7)))):
            image.save(self.work / (name + ".png"))
            with self.assertRaises(ValueError):
                composer.load_assets([{"id": name, "path": name + ".png"}], self.work)
        clipped = Image.new("RGBA", (7, 7))
        clipped.putpixel((0, 3), (250, 10, 20, 255))
        clipped.putpixel((3, 3), (250, 10, 20, 255))
        clipped.save(self.work / "edge.png")
        with self.assertRaisesRegex(ValueError, "touches its source edge"):
            composer.load_assets([{"id": "edge", "path": "edge.png"}], self.work)
        bad = dict(self.config, edge_margin=100)
        with self.assertRaisesRegex(ValueError, "Unknown config fields"):
            self.run_config("bad", bad)
        self.assertFalse((self.work / "bad").exists())

    def test_mixed_color_profiles_require_explicit_normalization(self):
        from PIL import ImageCms
        with Image.open(self.work / "motif.png") as image:
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            image.save(self.work / "profiled.png", icc_profile=profile)
        with self.assertRaisesRegex(ValueError, "different color profiles"):
            composer.load_assets([
                {"id": "a", "path": "motif.png"}, {"id": "b", "path": "profiled.png"},
            ], self.work)

    def test_cli_builds_qa_and_refuses_overwrite(self):
        path = self.work / "config.json"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        command = [sys.executable, "-B", str(SCRIPTS / "compose_pattern.py"), str(path), "--output-dir", str(self.work / "cli")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NOT CHECKED", result.stdout)
        tile_before = (self.work / "cli" / "tile.png").read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(tile_before, (self.work / "cli" / "tile.png").read_bytes())

    def test_object_spacing_survives_replay_and_checks_manual_placements(self):
        config = json.loads(json.dumps(self.config))
        config['width'], config['height'] = 110, 90
        config['object_spacing'] = {'min_gap_px': 3, 'alpha_threshold': 1}
        manifest, folder = self.run_config('spaced', config)
        self.assertTrue(manifest['object_spacing_verification']['transformed_alpha_masks_verified'])
        replay = json.loads((folder / 'replay-config.json').read_text(encoding='utf-8'))
        self.assertEqual(replay['object_spacing'], config['object_spacing'])
        path = composer.compose_pattern(folder / 'replay-config.json', self.work / 'spaced-replay')
        self.assertEqual((folder / 'tile.png').read_bytes(), (path.parent / 'tile.png').read_bytes())
        replay['placements'][1] = dict(replay['placements'][0])
        with self.assertRaisesRegex(ValueError, 'Object spacing violated at placement 1'):
            self.run_config('invalid-explicit', replay)
        self.assertFalse((self.work / 'invalid-explicit').exists())


class ObjectSpacingTests(unittest.TestCase):
    def assets(self, image):
        return {'shape': {'image': image}}

    def item(self, x, y, size, angle=0):
        return {'asset_id': 'shape', 'x': x, 'y': y, 'size_px': size, 'rotation_deg': angle}

    def render(self, image, items, period=(30, 30), gap=0):
        return composer.render_placements(self.assets(image), items, *period, (0, 0, 0, 0),
                                          {'min_gap_px': gap})

    def test_sizes_are_checked_even_when_center_distances_are_identical(self):
        image = Image.new('RGBA', (9, 9), (0, 0, 0, 255))
        self.render(image, [self.item(8, 15, 3), self.item(14, 15, 3)])
        with self.assertRaisesRegex(ValueError, 'placement 1'):
            self.render(image, [self.item(8, 15, 9), self.item(14, 15, 9)])
        # Opting out keeps the intended overlap behavior of dense prints.
        composer.render_placements(self.assets(image), [self.item(8, 15, 9), self.item(14, 15, 9)],
                                   30, 30, (0, 0, 0, 0))

    def test_edge_and_corner_neighbors_and_self_copies_are_checked(self):
        image = Image.new('RGBA', (5, 5), (0, 0, 0, 255))
        for first, second in [((0,10),(19,10)), ((10,0),(10,19)), ((0,0),(19,19))]:
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, 'placement 1'):
                    self.render(image, [self.item(*first, 5), self.item(*second, 5)], (20,20))
        with self.assertRaisesRegex(ValueError, 'placement 0'):
            self.render(image, [self.item(5,5,25)], (20,20))
        with self.assertRaisesRegex(ValueError, 'placement 0'):
            self.render(image, [self.item(5,5,18)], (20,20), gap=3)

    def test_alpha_holes_are_available_and_requested_gap_is_enforced(self):
        ring = Image.new('RGBA', (9,9))
        for y in range(9):
            for x in range(9):
                if x in (0,8) or y in (0,8):
                    ring.putpixel((x,y), (0,0,0,255))
        dot = Image.new('RGBA', (1,1), (0,0,0,255))
        space = composer.ObjectSpacing(30,30,{'min_gap_px':3})
        self.assertTrue(space.try_add(ring,15,15))
        self.assertTrue(space.try_add(dot,15,15))
        space = composer.ObjectSpacing(30,30,{'min_gap_px':4})
        self.assertTrue(space.try_add(ring,15,15))
        self.assertFalse(space.try_add(dot,15,15))

    def test_rotated_and_scaled_generated_shapes_match_independent_pixel_spacing(self):
        image = Image.new('RGBA', (13,5), (0,0,0,255))
        assets = self.assets(image)
        layout = {'count':16, 'size_px':[7,15], 'rotation_deg':[-80,80], 'max_attempts':3000}
        width, height, gap = 70, 61, 3
        items = composer.generate_placements(layout,117,width,height,assets,{'min_gap_px':gap})
        occupied = set()
        for item in items:
            motif = composer.prepare_motif(image,item['size_px'],item['rotation_deg'])
            alpha = motif.getchannel('A')
            left, top = item['x']-motif.width//2, item['y']-motif.height//2
            pixels = {(left+x,top+y) for y in range(motif.height) for x in range(motif.width)
                      if alpha.getpixel((x,y)) > 0}
            expanded = {((x+dx)%width,(y+dy)%height) for x,y in pixels
                        for dx in range(-gap,gap+1) for dy in range(-gap,gap+1)}
            self.assertFalse(expanded & occupied)
            occupied.update((x%width,y%height) for x,y in pixels)
        # Re-running with the same geometry and seed gives the same accepted layout.
        self.assertEqual(items, composer.generate_placements(layout,117,width,height,assets,{'min_gap_px':gap}))

    def test_rotation_is_part_of_the_silhouette_check(self):
        image = Image.new('RGBA', (9,1), (0,0,0,255))
        self.render(image,[self.item(10,10,9),self.item(10,13,9)])
        with self.assertRaisesRegex(ValueError,'placement 1'):
            self.render(image,[self.item(10,10,9,90),self.item(10,13,9,90)])

    def test_alpha_threshold_is_explicit_and_validated(self):
        shape = Image.new('RGBA',(7,7),(0,0,0,1))
        shape.putpixel((3,3),(0,0,0,255))
        space = composer.ObjectSpacing(30,30,{'min_gap_px':0})
        self.assertTrue(space.try_add(shape,10,10))
        self.assertFalse(space.try_add(shape,13,10))
        space = composer.ObjectSpacing(30,30,{'min_gap_px':0,'alpha_threshold':128})
        self.assertTrue(space.try_add(shape,10,10))
        self.assertTrue(space.try_add(shape,13,10))
        for spec in [{'min_gap_px':-1},{'min_gap_px':1.5},{'min_gap_px':True},
                     {'min_gap_px':0,'alpha_threshold':256},{'min_gap_px':0,'extra':1}]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                composer.ObjectSpacing(30,30,spec)

    def test_impossible_layout_fails_without_reducing_count(self):
        image = Image.new('RGBA',(5,5),(0,0,0,255))
        layout = {'count':2,'size_px':[5,5],'max_attempts':10}
        with self.assertRaisesRegex(ValueError,'Placed only 1 of 2'):
            composer.generate_placements(layout,1,5,5,self.assets(image),{'min_gap_px':0})

    def test_square_clearance_expansion_includes_exact_pixel_neighborhoods(self):
        mask = Image.new('L',(9,7),0)
        points = [(0,0),(8,6),(3,4)]
        for point in points:
            mask.putpixel(point,255)
        for gap in (0,1,2,3,7,12):
            with self.subTest(gap=gap):
                actual = composer.expand_square_mask(mask,gap)
                expected = {(x+gap+dx,y+gap+dy) for x,y in points
                            for dx in range(-gap,gap+1) for dy in range(-gap,gap+1)}
                pixels = {(x,y) for y in range(actual.height) for x in range(actual.width)
                          if actual.getpixel((x,y)) != 0}
                self.assertEqual(pixels,expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
