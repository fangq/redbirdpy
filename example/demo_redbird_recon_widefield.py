"""
Redbird - Wide-field reconstruction example matching MATLAB
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redbirdpy as rb
import iso2mesh as i2m
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Generate Source/Detector Patterns - MATCHING MATLAB
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# MATLAB: srcpattern = diag(ones(1, 16)) + diag(ones(1, 15), -1);
srcpattern = np.diag(np.ones(16)) + np.diag(np.ones(15), -1)
srcpattern[0, -1] = 1  # srcpattern(1, end) = 1

# MATLAB: srcpattern = permute(repmat(srcpattern, [1, 1, 16]), [2 3 1]);
srcpattern = np.tile(srcpattern[:, :, np.newaxis], (1, 1, 16))
srcpattern = np.transpose(srcpattern, (1, 2, 0))

# MATLAB: srcpattern = cat(3, srcpattern, permute(srcpattern, [2 1 3]));
srcpattern = np.concatenate([srcpattern, np.transpose(srcpattern, (1, 0, 2))], axis=2)
detpattern = srcpattern.copy()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Prepare simulation input
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s0 = [40, 40, 20]
s2 = [90, 20, 20]

nobbx, fcbbx, _ = i2m.meshabox([0, 0, 0], [120, 60, 40], 4)
nosp, fcsp, _ = i2m.meshasphere(s0, 5, 1)
nosp2, fcsp2, _ = i2m.meshasphere(s2, 7.5, 1)

no, fc = i2m.mergemesh(nobbx, fcbbx[:, :3], nosp, fcsp[:, :3])
no, fc = i2m.mergemesh(no, fc[:, :3], nosp2, fcsp2[:, :3])

node, elem, _ = i2m.s2m(no, fc[:, :3], 1, 20, "tetgen", [[11, 1, 1], s0, s2])

cfg0 = {
    "node": node,
    "elem": elem[:, :4],
    "seg": elem[:, 4].astype(int),
    "srctype": "pattern",
    "srcpos": np.array([[10, 10, 0]]),
    "srcparam1": [100, 0, 0, 0],
    "srcparam2": [0, 40, 0, 0],
    "srcdir": np.array([[0, 0, 1]]),
    "srcpattern": srcpattern,
    "dettype": "pattern",
    "detpos": np.array([[10, 10, 40]]),
    "detparam1": [100, 0, 0, 0],
    "detparam2": [0, 40, 0, 0],
    "detdir": np.array([[0, 0, -1]]),
    "detpattern": detpattern,
    "prop": np.array(
        [
            [0, 0, 1, 1],
            [0.008, 1, 0, 1.37],
            [0.032, 1, 0, 1.37],
            [0.032, 1, 0, 1.37],
        ]
    ),
    "omega": 0,
}

cfg0 = rb.meshprep(cfg0)[0]

if "widesrc" in cfg0 and cfg0["widesrc"].size > 0:
    ws = cfg0["widesrc"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Run forward simulation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

detphi0 = rb.run(cfg0)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Create reconstruction mesh (matching MATLAB)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# MATLAB: cfg = rbsetmesh(cfg, cfg.node, cfg.elem, [0 0 1 1; 0.010 1 0 1.37], ones(size(cfg.node, 1), 1));
node_recon, _, elem_recon = i2m.meshabox([0, 0, 0], [120, 60, 40], 4)

cfg = {
    "node": node_recon,
    "elem": elem_recon[:, :4],
    "seg": np.ones(elem_recon.shape[0], dtype=int),
    "srctype": "pattern",
    "srcpos": np.array([[10, 10, 0]]),
    "srcparam1": [100, 0, 0, 0],
    "srcparam2": [0, 40, 0, 0],
    "srcdir": np.array([[0, 0, 1]]),
    "srcpattern": srcpattern,
    "dettype": "pattern",
    "detpos": np.array([[10, 10, 40]]),
    "detparam1": [100, 0, 0, 0],
    "detparam2": [0, 40, 0, 0],
    "detdir": np.array([[0, 0, -1]]),
    "detpattern": detpattern,
    "prop": np.array(
        [
            [0, 0, 1, 1],
            [0.010, 1, 0, 1.37],
        ]
    ),
    "omega": 0,
}

cfg = rb.meshprep(cfg)[0]
sd = rb.sdmap(cfg)

# Run forward on homogeneous
detphi_homog = rb.run(cfg)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Setup dual-mesh reconstruction
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

recon = {}
recon["node"], _, recon["elem"] = i2m.meshabox([0, 0, 0], [120, 60, 40], 15)
recon["mapid"], recon["mapweight"] = i2m.tsearchn(
    recon["node"], recon["elem"], cfg["node"]
)
recon["bulk"] = {"mua": 0.008, "musp": 1}  # Match MATLAB

nn_recon = recon["node"].shape[0]
recon["prop"] = np.zeros((nn_recon, 4))
recon["prop"][:, 0] = 0.010
recon["prop"][:, 1] = 1.0
recon["prop"][:, 2] = 0.0
recon["prop"][:, 3] = 1.37

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Run reconstruction
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

newrecon, resid = rb.run(cfg, recon, detphi0, lambda_=1e-4, maxiter=10)[:2]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Plot results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cutpos, cutval, facedata = i2m.qmeshcut(
    recon["elem"][:, :4], recon["node"][:, :3], newrecon["prop"][:, 0], "z = 20"
)[:3]
hh = i2m.plotmesh(
    np.c_[cutpos, np.log10(np.abs(cutval) + 1e-20)],
    facedata.tolist(),
    subplot=121,
    hold="on",
)
cutpos, cutval, facedata = i2m.qmeshcut(
    recon["elem"][:, :4], recon["node"][:, :3], newrecon["prop"][:, 0], "x = 90"
)[:3]
i2m.plotmesh(
    np.c_[cutpos, np.log10(np.abs(cutval) + 1e-20)],
    facedata.tolist(),
    subplot=122,
    parent=hh,
)

plt.show(block=(len(sys.argv) == 1))
