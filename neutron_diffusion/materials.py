import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Material1G:
    name: str
    Sigma_a: float
    Sigma_f: float
    nu: float
    D: float
    Sigma_s: float = 0.0

    @property
    def nuSigma_f(self) -> float:
        return self.nu * self.Sigma_f

    @property
    def Sigma_tr(self) -> float:
        return self.Sigma_a + self.Sigma_s


@dataclass
class Material2G:
    name: str
    D1: float
    Sigma_a1: float
    Sigma_f1: float
    nu1: float
    D2: float
    Sigma_a2: float
    Sigma_f2: float
    nu2: float
    Sigma_s12: float

    @property
    def nuSigma_f1(self) -> float:
        return self.nu1 * self.Sigma_f1

    @property
    def nuSigma_f2(self) -> float:
        return self.nu2 * self.Sigma_f2

    @property
    def Sigma_tr1(self) -> float:
        return self.Sigma_a1 + self.Sigma_s12

    @property
    def Sigma_tr2(self) -> float:
        return self.Sigma_a2


PRESET_MATERIALS_1G: Dict[str, Material1G] = {
    "UO2燃料(3%)": Material1G(
        name="UO2燃料(3%)",
        Sigma_a=0.10,
        Sigma_f=0.06,
        nu=2.42,
        D=1.2,
        Sigma_s=0.4
    ),
    "轻水慢化剂": Material1G(
        name="轻水慢化剂",
        Sigma_a=0.02,
        Sigma_f=0.0,
        nu=2.4,
        D=0.8,
        Sigma_s=3.0
    ),
    "石墨反射层": Material1G(
        name="石墨反射层",
        Sigma_a=0.003,
        Sigma_f=0.0,
        nu=2.4,
        D=1.5,
        Sigma_s=0.38
    ),
    "B4C控制棒": Material1G(
        name="B4C控制棒",
        Sigma_a=2.0,
        Sigma_f=0.0,
        nu=0.0,
        D=0.5,
        Sigma_s=0.2
    ),
    "真空": Material1G(
        name="真空",
        Sigma_a=1e-10,
        Sigma_f=0.0,
        nu=0.0,
        D=1e6,
        Sigma_s=0.0
    ),
}

PRESET_MATERIALS_2G: Dict[str, Material2G] = {
    "UO2燃料(3%)": Material2G(
        name="UO2燃料(3%)",
        D1=1.5,
        Sigma_a1=0.01,
        Sigma_f1=0.005,
        nu1=2.42,
        D2=0.8,
        Sigma_a2=0.12,
        Sigma_f2=0.08,
        nu2=2.42,
        Sigma_s12=0.08
    ),
    "轻水慢化剂": Material2G(
        name="轻水慢化剂",
        D1=1.2,
        Sigma_a1=0.001,
        Sigma_f1=0.0,
        nu1=2.4,
        D2=0.5,
        Sigma_a2=0.02,
        Sigma_f2=0.0,
        nu2=2.4,
        Sigma_s12=0.5
    ),
    "石墨反射层": Material2G(
        name="石墨反射层",
        D1=1.8,
        Sigma_a1=0.0005,
        Sigma_f1=0.0,
        nu1=2.4,
        D2=1.0,
        Sigma_a2=0.004,
        Sigma_f2=0.0,
        nu2=2.4,
        Sigma_s12=0.08
    ),
    "B4C控制棒": Material2G(
        name="B4C控制棒",
        D1=0.8,
        Sigma_a1=0.5,
        Sigma_f1=0.0,
        nu1=0.0,
        D2=0.3,
        Sigma_a2=2.5,
        Sigma_f2=0.0,
        nu2=0.0,
        Sigma_s12=0.01
    ),
    "真空": Material2G(
        name="真空",
        D1=1e6,
        Sigma_a1=1e-10,
        Sigma_f1=0.0,
        nu1=0.0,
        D2=1e6,
        Sigma_a2=1e-10,
        Sigma_f2=0.0,
        nu2=0.0,
        Sigma_s12=0.0
    ),
}


def get_preset_1g(name: str) -> Material1G:
    if name in PRESET_MATERIALS_1G:
        return PRESET_MATERIALS_1G[name]
    raise ValueError(f"未知材料预设: {name}")


def get_preset_2g(name: str) -> Material2G:
    if name in PRESET_MATERIALS_2G:
        return PRESET_MATERIALS_2G[name]
    raise ValueError(f"未知材料预设: {name}")
