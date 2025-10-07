import argparse, re, cv2, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# --- camera intrinsics / distortion (same as Q2-1) ---
K = np.array([[1868.27, 0.00,  540.00],
              [0.00,    1869.18,960.00],
              [0.00,    0.00,   1.00 ]], dtype=np.float64)
DIST = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float64)  # [k1,k2,p1,p2]

def frame_idx_from_name(name: str) -> int:
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1

def load_validation_poses_gt(images_pkl: str):
    """Load GT W2C poses from images.pkl"""
    df = pd.read_pickle(images_pkl)
    valid = df[df["NAME"].str.startswith("valid_img")].copy()
    valid["FRAME_IDX"] = valid["NAME"].apply(frame_idx_from_name)
    valid = valid.sort_values("FRAME_IDX")
    poses = []
    for _, row in valid.iterrows():
        q_xyzw = row[["QX","QY","QZ","QW"]].to_numpy(dtype=np.float64)  # already xyzw
        t_w2c  = row[["TX","TY","TZ"]].to_numpy(dtype=np.float64)
        poses.append({"name": str(row["NAME"]), "q_xyzw": q_xyzw, "t_w2c": t_w2c})
    return poses

def load_validation_poses_est(csv_path: str, images_dir: Path):
    """
    Load estimated W2C poses from CSV (case-insensitive headers).
    Expected logical columns (any case in the CSV is OK):
      NAME,qx,qy,qz,qw,tx,ty,tz
    Returns only rows whose images exist in images_dir and start with 'valid_img'.
    """
    df = pd.read_csv(csv_path)

    # Build case-insensitive map: 'qx' -> actual column name in CSV
    cols_lower = {c.lower(): c for c in df.columns}

    required_lower = {"name","qx","qy","qz","qw","tx","ty","tz"}
    missing = [k for k in required_lower if k not in cols_lower]
    if missing:
        raise ValueError(
            f"CSV must have columns (any case): {sorted(required_lower)} "
            f"(got {list(df.columns)})"
        )

    # Rename to canonical names used below
    rename_map = {
        cols_lower["name"]: "NAME",
        cols_lower["qx"]:   "qx",
        cols_lower["qy"]:   "qy",
        cols_lower["qz"]:   "qz",
        cols_lower["qw"]:   "qw",
        cols_lower["tx"]:   "tx",
        cols_lower["ty"]:   "ty",
        cols_lower["tz"]:   "tz",
    }
    df = df.rename(columns=rename_map)

    # Keep only validation frames that exist on disk
    df = df[df["NAME"].astype(str).str.startswith("valid_img")].copy()
    df["FRAME_IDX"] = df["NAME"].apply(frame_idx_from_name)
    df = df.sort_values("FRAME_IDX")

    poses = []
    for _, row in df.iterrows():
        name = str(row["NAME"])
        img_path = images_dir / name
        if not img_path.exists():
            continue
        q_xyzw = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
        t_w2c  = np.array([row["tx"], row["ty"], row["tz"]], dtype=np.float64)
        poses.append({"name": name, "q_xyzw": q_xyzw, "t_w2c": t_w2c})

    if len(poses) == 0:
        raise ValueError("No usable estimated poses found (check NAME paths and CSV rows).")
    return poses


def load_cube_transform(npy_path):
    # 3x4 world transform [S*R | t] saved by transform_cube.py
    T = np.load(npy_path)  # shape (3,4)
    Rw = T[:, :3]
    tw = T[:, 3]
    return Rw, tw

def sample_cube_points(res=36):
    """Sample unit-cube surfaces as colored points."""
    u = np.linspace(0, 1, res)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    uu = uu.reshape(-1); vv = vv.reshape(-1)
    faces = [
        np.stack([np.zeros_like(uu), uu, vv], axis=1),  # x=0
        np.stack([np.ones_like(uu),  uu, vv], axis=1),  # x=1
        np.stack([uu, np.zeros_like(uu), vv], axis=1),  # y=0
        np.stack([uu, np.ones_like(uu),  vv], axis=1),  # y=1
        np.stack([uu, vv, np.zeros_like(uu)], axis=1),  # z=0
        np.stack([uu, vv, np.ones_like(uu)],  axis=1),  # z=1
    ]
    X = np.concatenate(faces, axis=0).astype(np.float64)
    face_ids = np.concatenate([np.full(res*res, i) for i in range(6)], axis=0)
    return X, face_ids

def apply_world_transform(X_unit, Rw, tw):
    return (Rw @ X_unit.T).T + tw.reshape(1,3)

def painter_sort_and_project(X_world, face_ids, R_w2c, t_w2c):
    # Camera-space depth (+Z forward)
    X_cam = (R_w2c @ X_world.T).T + t_w2c.reshape(1,3)
    z = X_cam[:, 2]
    keep = z > 1e-6
    if not np.any(keep):
        return np.empty((0,2), np.float32), np.empty((0,3), np.uint8)

    X_world = X_world[keep]; face_ids = face_ids[keep]; z = z[keep]
    order = np.argsort(-z)  # far -> near
    X_sorted = X_world[order]
    fid_sorted = face_ids[order]

    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = t_w2c.reshape(3,1)
    img_pts, _ = cv2.projectPoints(X_sorted, rvec, tvec, K, DIST)
    img_pts = img_pts.reshape(-1,2).astype(np.float32)

    face_bgr = np.array([
        [  0,   0, 255],  # red
        [  0, 255,   0],  # green
        [255,   0,   0],  # blue
        [255, 255,   0],  # cyan
        [255,   0, 255],  # magenta
        [  0, 255, 255],  # yellow
    ], dtype=np.uint8)
    colors = face_bgr[fid_sorted]
    return img_pts, colors

def main(args):
    images_dir = Path("data/frames")

    # --- pick poses: estimated (CSV) or GT (images.pkl) ---
    if args.use_est:
        poses = load_validation_poses_est(args.use_est, images_dir)
        print(f"[INFO] Using ESTIMATED poses from: {args.use_est}  (#frames: {len(poses)})")
    else:
        poses = load_validation_poses_gt("data/images.pkl")
        print(f"[INFO] Using GROUND-TRUTH poses from data/images.pkl  (#frames: {len(poses)})")

    # --- cube world transform ---
    Rw, tw = load_cube_transform(args.cube_transform)

    # --- sample cube points ---
    X_unit, face_ids = sample_cube_points(res=args.res)

    # --- prepare writer ---
    first = cv2.imread(str(images_dir / poses[0]["name"]))
    if first is None:
        raise FileNotFoundError(f"Cannot read first frame: {images_dir / poses[0]['name']}")
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    for p in poses:
        img_path = images_dir / p["name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # world cube points
        Xw = apply_world_transform(X_unit, Rw, tw)

        # W2C from quat (xyzw) + t
        R_w2c = R.from_quat(p["q_xyzw"]).as_matrix()
        t_w2c = p["t_w2c"].astype(np.float64)

        # painter’s algorithm
        pts_2d, cols = painter_sort_and_project(Xw, face_ids, R_w2c, t_w2c)

        # draw far->near (already sorted)
        r = args.pt_radius
        for (u,v), c in zip(pts_2d, cols):
            uu = int(round(float(u))); vv = int(round(float(v)))
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(img, (uu, vv), r, color=tuple(int(x) for x in c.tolist()),
                           thickness=-1, lineType=cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"[OK] Saved AR video to: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube_transform", type=str, default="cube_transform_mat.npy",
                        help="3x4 transform saved by transform_cube.py")
    parser.add_argument("--out", type=str, default="ar_valid.mp4", help="output mp4")
    parser.add_argument("--fps", type=int, default=15, help="video fps")
    parser.add_argument("--res", type=int, default=36, help="points per side (face has res*res points)")
    parser.add_argument("--pt_radius", type=int, default=1, help="drawn point radius (pixels)")
    parser.add_argument("--use_est", type=str, default=None,
                        help="CSV path with estimated W2C poses (columns: NAME,qx,qy,qz,qw,tx,ty,tz)")
    args = parser.parse_args()
    main(args)
