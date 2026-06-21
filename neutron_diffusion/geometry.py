import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .materials import Material1G, Material2G, get_preset_1g, get_preset_2g


@dataclass
class BoundaryCondition:
    left: str = "vacuum"
    right: str = "vacuum"
    top: str = "vacuum"
    bottom: str = "vacuum"


@dataclass
class Region1D:
    x_start: float
    x_end: float
    material_name: str


@dataclass
class Geometry1D:
    regions: List[Region1D]
    dx: float = 1.0
    bc: BoundaryCondition = field(default_factory=BoundaryCondition)
    material_mode: str = "1g"

    @property
    def total_width(self) -> float:
        if not self.regions:
            return 0.0
        return self.regions[-1].x_end - self.regions[0].x_start

    @property
    def x_min(self) -> float:
        return self.regions[0].x_start if self.regions else 0.0

    @property
    def x_max(self) -> float:
        return self.regions[-1].x_end if self.regions else 0.0

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        x_nodes = np.arange(self.x_min, self.x_max + self.dx / 2, self.dx)
        x_centers = (x_nodes[:-1] + x_nodes[1:]) / 2.0
        n_cells = len(x_centers)
        material_names = []
        for i in range(n_cells):
            xc = x_centers[i]
            mat_name = self._find_material(xc)
            material_names.append(mat_name)
        return x_nodes, x_centers, material_names

    def _find_material(self, x: float) -> str:
        for r in self.regions:
            if r.x_start <= x <= r.x_end:
                return r.material_name
        return self.regions[-1].material_name if self.regions else "真空"

    def get_materials_1g(self) -> List[Material1G]:
        _, _, material_names = self.build_mesh()
        return [get_preset_1g(m) for m in material_names]

    def get_materials_2g(self) -> List[Material2G]:
        _, _, material_names = self.build_mesh()
        return [get_preset_2g(m) for m in material_names]

    def get_extrapolation_distances_1g(self) -> Tuple[float, float]:
        materials = self.get_materials_1g()
        d_left = 0.7104 / materials[0].Sigma_tr if self.bc.left == "vacuum" else 0.0
        d_right = 0.7104 / materials[-1].Sigma_tr if self.bc.right == "vacuum" else 0.0
        return d_left, d_right

    def get_extrapolation_distances_2g(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        materials = self.get_materials_2g()
        if self.bc.left == "vacuum":
            d_left_1 = 0.7104 / materials[0].Sigma_tr1
            d_left_2 = 0.7104 / materials[0].Sigma_tr2
        else:
            d_left_1, d_left_2 = 0.0, 0.0
        if self.bc.right == "vacuum":
            d_right_1 = 0.7104 / materials[-1].Sigma_tr1
            d_right_2 = 0.7104 / materials[-1].Sigma_tr2
        else:
            d_right_1, d_right_2 = 0.0, 0.0
        return (d_left_1, d_left_2), (d_right_1, d_right_2)


@dataclass
class Region2D:
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    material_name: str


@dataclass
class Geometry2D:
    regions: List[Region2D]
    dx: float = 1.0
    dy: float = 1.0
    bc: BoundaryCondition = field(default_factory=BoundaryCondition)
    material_mode: str = "1g"

    @property
    def x_min(self) -> float:
        return min(r.x_start for r in self.regions) if self.regions else 0.0

    @property
    def x_max(self) -> float:
        return max(r.x_end for r in self.regions) if self.regions else 0.0

    @property
    def y_min(self) -> float:
        return min(r.y_start for r in self.regions) if self.regions else 0.0

    @property
    def y_max(self) -> float:
        return max(r.y_end for r in self.regions) if self.regions else 0.0

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
        x_nodes = np.arange(self.x_min, self.x_max + self.dx / 2, self.dx)
        y_nodes = np.arange(self.y_min, self.y_max + self.dy / 2, self.dy)
        x_centers = (x_nodes[:-1] + x_nodes[1:]) / 2.0
        y_centers = (y_nodes[:-1] + y_nodes[1:]) / 2.0
        nx = len(x_centers)
        ny = len(y_centers)
        material_names = [["" for _ in range(nx)] for _ in range(ny)]
        for j in range(ny):
            for i in range(nx):
                xc, yc = x_centers[i], y_centers[j]
                material_names[j][i] = self._find_material(xc, yc)
        return x_nodes, y_nodes, x_centers, y_centers, material_names

    def _find_material(self, x: float, y: float) -> str:
        for r in self.regions:
            if r.x_start <= x <= r.x_end and r.y_start <= y <= r.y_end:
                return r.material_name
        return "真空"

    def get_materials_1g(self) -> List[List[Material1G]]:
        _, _, _, _, material_names = self.build_mesh()
        return [[get_preset_1g(m) for m in row] for row in material_names]

    def get_materials_2g(self) -> List[List[Material2G]]:
        _, _, _, _, material_names = self.build_mesh()
        return [[get_preset_2g(m) for m in row] for row in material_names]

    def get_extrapolation_distances_1g(self) -> Tuple[float, float, float, float]:
        materials = self.get_materials_1g()
        ny, nx = len(materials), len(materials[0])
        d_left = 0.7104 / materials[ny // 2][0].Sigma_tr if self.bc.left == "vacuum" else 0.0
        d_right = 0.7104 / materials[ny // 2][nx - 1].Sigma_tr if self.bc.right == "vacuum" else 0.0
        d_bottom = 0.7104 / materials[0][nx // 2].Sigma_tr if self.bc.bottom == "vacuum" else 0.0
        d_top = 0.7104 / materials[ny - 1][nx // 2].Sigma_tr if self.bc.top == "vacuum" else 0.0
        return d_left, d_right, d_bottom, d_top

    def get_extrapolation_distances_2g(self) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        materials = self.get_materials_2g()
        ny, nx = len(materials), len(materials[0])
        if self.bc.left == "vacuum":
            d_left_1 = 0.7104 / materials[ny // 2][0].Sigma_tr1
            d_left_2 = 0.7104 / materials[ny // 2][0].Sigma_tr2
        else:
            d_left_1, d_left_2 = 0.0, 0.0
        if self.bc.right == "vacuum":
            d_right_1 = 0.7104 / materials[ny // 2][nx - 1].Sigma_tr1
            d_right_2 = 0.7104 / materials[ny // 2][nx - 1].Sigma_tr2
        else:
            d_right_1, d_right_2 = 0.0, 0.0
        if self.bc.bottom == "vacuum":
            d_bottom_1 = 0.7104 / materials[0][nx // 2].Sigma_tr1
            d_bottom_2 = 0.7104 / materials[0][nx // 2].Sigma_tr2
        else:
            d_bottom_1, d_bottom_2 = 0.0, 0.0
        if self.bc.top == "vacuum":
            d_top_1 = 0.7104 / materials[ny - 1][nx // 2].Sigma_tr1
            d_top_2 = 0.7104 / materials[ny - 1][nx // 2].Sigma_tr2
        else:
            d_top_1, d_top_2 = 0.0, 0.0
        dist1 = (d_left_1, d_right_1, d_bottom_1, d_top_1)
        dist2 = (d_left_2, d_right_2, d_bottom_2, d_top_2)
        return dist1, dist2
