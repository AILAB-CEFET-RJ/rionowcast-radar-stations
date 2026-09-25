from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import shapefile

from nowcasting.ibge_basemap import build_municipal_basemap


class IbgeBasemapTests(unittest.TestCase):
    def test_filters_municipalities_to_radar_extent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shapefile_path = root / "municipios"
            with shapefile.Writer(str(shapefile_path)) as writer:
                writer.field("CD_MUN", "C")
                writer.field("NM_MUN", "C")
                writer.poly([[[-43.4, -23.1], [-43.4, -22.8], [-43.0, -22.8], [-43.0, -23.1], [-43.4, -23.1]]])
                writer.record("1", "Dentro")
                writer.poly([[[-45.4, -25.1], [-45.4, -24.8], [-45.0, -24.8], [-45.0, -25.1], [-45.4, -25.1]]])
                writer.record("2", "Fora")
            grid_path = root / "grid.npz"
            np.savez(
                grid_path,
                lat=np.array([[-23.2, -23.2], [-22.7, -22.7]]),
                lon=np.array([[-43.5, -42.9], [-43.5, -42.9]]),
            )
            output_path = root / "municipios.geojson"

            summary = build_municipal_basemap(shapefile_path.with_suffix(".shp"), grid_path, output_path)
            geojson = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["municipalities"], 1)
        self.assertEqual(geojson["features"][0]["properties"]["NM_MUN"], "Dentro")


if __name__ == "__main__":
    unittest.main()
