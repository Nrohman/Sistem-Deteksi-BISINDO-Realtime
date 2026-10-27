import os
os.environ['KERAS_HOME'] = os.path.abspath('.keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np

df = pd.read_csv('data/bisindo_Landmarks.csv')
features = df.drop('label', axis=1).values
valid_mask = (np.abs(features).sum(axis=1) > 0)
df_clean = df[valid_mask]

for letter in ['A', 'Y', 'J', 'K', 'P']:
    sub = df_clean[df_clean['label'] == letter].drop('label', axis=1).values
    print(f"\n==================== Letter {letter} (N={len(sub)}) ====================")
    for h_name, start in [('Left', 0), ('Right', 63)]:
        h_vals = sub[:, start:start+63]
        active = (np.abs(h_vals).sum(axis=1) > 0)
        print(f"  {h_name} hand active: {active.sum()}/{len(sub)} ({active.mean()*100:.1f}%)")
        if active.sum() > 0:
            h_active = h_vals[active].reshape(-1, 21, 3)
            # Check bounding box (min/max x, y)
            min_xy = h_active[:, :, :2].min(axis=1).mean(axis=0)
            max_xy = h_active[:, :, :2].max(axis=1).mean(axis=0)
            print(f"    Avg BBox [xmin, ymin] -> [xmax, ymax]: [{min_xy[0]:.2f}, {min_xy[1]:.2f}] -> [{max_xy[0]:.2f}, {max_xy[1]:.2f}]")
            # Check extended fingers: distance from wrist (0) to fingertip vs mcp
            # thumb tip (4) vs thumb mcp (2)
            # index tip (8) vs index mcp (5)
            # middle tip (12) vs middle mcp (9)
            # ring tip (16) vs ring mcp (13)
            # pinky tip (20) vs pinky mcp (17)
            wrist = h_active[:, 0, :]
            tips = [4, 8, 12, 16, 20]
            mcps = [2, 5, 9, 13, 17]
            extended = []
            for tip, mcp in zip(tips, mcps):
                d_tip = np.linalg.norm(h_active[:, tip, :] - wrist, axis=1)
                d_mcp = np.linalg.norm(h_active[:, mcp, :] - wrist, axis=1)
                extended.append((d_tip > d_mcp).mean())
            print(f"    Fingers extended (Thumb, Index, Mid, Ring, Pinky): {[round(x, 2) for x in extended]}")
