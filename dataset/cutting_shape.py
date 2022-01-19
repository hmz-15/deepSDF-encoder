import os
import numpy as np
from utils import motion_util
import math
import glob


class CuttingShape:
    """
    Use cutting shape data (normalized)
    """
    def __init__(self, cutting_shape_path):
        self.data_sources = glob.glob(cutting_shape_path + "/*.obj")

    def __len__(self):
        return len(self.data_sources)

    @staticmethod
    def _equidist_point_on_sphere(samples):
        points = []
        phi = math.pi * (3. - math.sqrt(5.))

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append((x, y, z))

        return np.asarray(points)

    def get_source(self, data_id):
        return self.data_sources[data_id]

    def __getitem__(self, idx):
        data_source = self.data_sources[idx]
        vp_camera = self._equidist_point_on_sphere(300)
        camera_ext = []
        for camera_i in range(vp_camera.shape[0]):
            iso = motion_util.Isometry.look_at(vp_camera[camera_i], np.zeros(3,))
            camera_ext.append(iso)
        camera_int = [0.8, 0.0, 2.5]  # (window-size-half, z-min, z-max) under ortho-proj.

        return data_source, [camera_int, camera_ext], None, 1.0

    def clean(self, data_id):
        pass
