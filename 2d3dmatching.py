from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
from tqdm import tqdm
import open3d as o3d
import re
import torch

def frame_idx_from_name(name: str) -> int:
    # 例如 valid_img260.jpg → 260；train_img552.jpg → 552
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1

np.random.seed(1428)
random.seed(1428)

def average(x):
    return list(np.mean(x, axis=0))

def average_desc(train_df, points3D_df):
    # 針對同一 3D 點（POINT_ID）的多個描述子取平均，建立「模型資料庫」
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average).reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def _build_matcher():
    # 128D float 描述子 → 用 FLANN KD-Tree，比 BF 快
    index_params = dict(algorithm=1, trees=8)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)

import torch

def pnpsolver(query, model, cameraMatrix=0, distortion=0):
    kp_query, desc_query = query            # (N,2), (N,128) float32
    kp_model, desc_model = model            # (M,3), (M,128) float32

    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float64)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float64)

    if len(desc_query) == 0 or len(desc_model) == 0:
        return False, None, None, None

    # ---- GPU kNN: cosine 相似 + ratio test ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.from_numpy(desc_query).to(device)  # [N,128]
    d = torch.from_numpy(desc_model).to(device)  # [M,128]

    # 先 L2 normalize → 用內積當 cosine 相似
    q = torch.nn.functional.normalize(q, dim=1)
    d = torch.nn.functional.normalize(d, dim=1)

    # 依記憶體情況分批算相似度（避免 N x M 全矩陣太大）
    topk = 2
    ratio = 0.8
    good_pts2d = []
    good_pts3d = []

    B = 4096  # 批大小，可依顯存調大/調小
    with torch.no_grad():
        for s in range(0, q.size(0), B):
            qe = q[s:s+B]                    # [b,128]
            # 相似度：qe @ d^T  → [b,M]
            sim = torch.matmul(qe, d.t())
            # 取 top-2（最大相似度；若你要 L2 最小距離，改用 torch.cdist 再取最小）
            vals, idxs = torch.topk(sim, k=topk, dim=1, largest=True, sorted=True)
            # ratio test（相似度越大越好 → 轉成距離概念為 vals[:,0] / vals[:,1]）
            # 這裡直接用相似度版 ratio：要求 vals0 > ratio * vals1
            mask = vals[:, 0] > ratio * vals[:, 1]
            if mask.any():
                sel_q = (s + torch.nonzero(mask, as_tuple=False).squeeze(1)).cpu().numpy()
                sel_m = idxs[mask, 0].cpu().numpy()
                good_pts2d.extend(kp_query[sel_q])
                good_pts3d.extend(kp_model[sel_m])

    if len(good_pts2d) < 6:
        return False, None, None, None

    pts2d = np.asarray(good_pts2d, dtype=np.float64)
    pts3d = np.asarray(good_pts3d, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d,
        imagePoints=pts2d,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=2000
    )
    if not ok:
        return False, None, None, None
    return True, rvec, tvec, inliers


def rotation_error(R_est_quat_xyzw, R_gt_quat_xyzw):
    """
    旋轉誤差 = 相對旋轉的軸角角度（度）
    1) 建 R_est, R_gt 的 Rotation
    2) R_rel = R_est * R_gt.inv()
    3) angle = ||rotvec||（再轉度）
    SciPy Rotation 預期四元數順序為 [x,y,z,w]；講義資料 image.pkl 的宣告為 (QW,QX,QY,QZ)，
    你若直接從 DataFrame 讀到 [QX,QY,QZ,QW]，那就正好是 [x,y,z,w] 可直接餵入。:contentReference[oaicite:3]{index=3}
    """
    r_est = R.from_quat(R_est_quat_xyzw)
    r_gt  = R.from_quat(R_gt_quat_xyzw)
    r_rel = r_est * r_gt.inv()
    ang_rad = np.linalg.norm(r_rel.as_rotvec())
    return np.degrees(ang_rad)

def translation_error(t_est, t_gt):
    """ 平移誤差 = ||t_est - t_gt||_2 """
    return float(np.linalg.norm(t_est.reshape(-1) - t_gt.reshape(-1)))

def _make_frustum_lineset(T_wc, scale=0.5):
    """
    用四角錐表示相機姿態（Open3D LineSet）
    頂點：相機中心 C_w；底面法向量：相機 -Z（視線方向）
    """
    # 基本相機金字塔（在相機座標系）：Z 前方為 -Z
    C = np.array([0, 0, 0, 1.0])
    a = np.array([ scale,  scale, -1.0, 1.0])
    b = np.array([ scale, -scale, -1.0, 1.0])
    c = np.array([-scale, -scale, -1.0, 1.0])
    d = np.array([-scale,  scale, -1.0, 1.0])
    pts_cam = np.stack([C, a, b, c, d], axis=0).T  # 4x5

    pts_world = (T_wc @ pts_cam).T[:, :3]  # 5x3
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    return ls

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    import open3d as o3d
    import numpy as np

    geoms = []

    # === 1) 建立點雲 (XYZ + RGB) ===
    # 位置
    if "XYZ" in points3D_df.columns:
        pts = np.vstack(points3D_df["XYZ"].to_list()).astype(np.float64)   # (N,3)
    else:
        pts = points3D_df[["X","Y","Z"]].values.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 顏色（支援兩種欄位型式）
    if "RGB" in points3D_df.columns:
        rgb = np.vstack(points3D_df["RGB"].to_list()).astype(np.float32)   # (N,3), 0~255
    elif set(["R","G","B"]).issubset(points3D_df.columns):
        rgb = points3D_df[["R","G","B"]].values.astype(np.float32)         # (N,3), 0~255
    else:
        # 沒有顏色就給個預設淡藍
        rgb = np.tile(np.array([[120, 180, 255]], dtype=np.float32), (pts.shape[0], 1))

    rgb = np.clip(rgb / 255.0, 0.0, 1.0)                                   # 轉 0~1
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # （可選）估計法向量：對點雲上色不是必須，但後續想做法向可視化或濾波會用到
    # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    geoms.append(pcd)

    # === 2) 相機四角錐 + 軌跡 ===
    def _make_frustum_lineset(T_wc, size=0.2, color=(1.0, 0.0, 0.0)):
        """
        等比例縮放的相機四角錐（+Z 朝前）
        - size: 全尺寸尺度（同時影響 XY與Z）
        """
        import numpy as np
        import open3d as o3d

        half = 0.5 * size    # 底面半寬（等比）
        depth = 1.0 * size   # 錐體深度（等比）

        C = np.array([0,    0,     0,    1.0])
        a = np.array([ half, half, depth, 1.0])
        b = np.array([ half,-half, depth, 1.0])
        c = np.array([-half,-half, depth, 1.0])
        d = np.array([-half, half, depth, 1.0])

        pts_cam = np.stack([C, a, b, c, d], axis=0).T  # 4x5
        pts_world = (T_wc @ pts_cam).T[:, :3]
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_world)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
        )
        return ls



    traj_xyz = []
    for T in Camera2World_Transform_Matrixs:
        # 四角錐：用預設 scale=0.3、紅色
        geoms.append(_make_frustum_lineset(T, size=0.1, color=(1.0, 0.0, 0.0)))
        traj_xyz.append(T[:3, 3])

    if len(traj_xyz) >= 2:
        import numpy as np
        traj = o3d.geometry.LineSet()
        traj_pts = np.array(traj_xyz, dtype=np.float64)
        traj.points = o3d.utility.Vector3dVector(traj_pts)
        traj.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj_pts)-1)])
        # 軌跡著色為藍色
        traj.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(traj.lines), 1))
        )
        geoms.append(traj)

    # === 3) 用 Visualizer 調參：點大小、背景顏色、顯示座標軸 ===
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Relocalization: RGB Point Cloud + Camera Poses", width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    opt.point_size = 2.0  # 白底上點太粗會糊，略調小
    opt.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 讀資料（講義與骨架都要求用 pandas 讀 pickle）:contentReference[oaicite:5]{index=5}
    images_df     = pd.read_pickle("data/images.pkl")
    train_df      = pd.read_pickle("data/train.pkl")
    points3D_df   = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # 建立模型 3D 平均描述子庫
    desc_df   = average_desc(train_df, points3D_df)
    kp_model  = np.array(desc_df["XYZ"].to_list(), dtype=np.float32)              # (M,3)
    desc_model= np.array(desc_df["DESCRIPTORS"].to_list(), dtype=np.float32)      # (M,128)

    # 只跑驗證集影像（依檔名過濾 data/frames/valid_imgXXX.jpg）
    valid_mask = images_df["NAME"].str.startswith("valid_img")
    valid_df = images_df[valid_mask].copy()
    valid_df["FRAME_IDX"] = valid_df["NAME"].apply(frame_idx_from_name)
    valid_df = valid_df.sort_values("FRAME_IDX")  # 嚴格依檔名中的數字排序


    r_list, t_list = [], []
    rot_err_list, trans_err_list = [], []

    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        idx  = int(row["IMAGE_ID"])
        name = str(row["NAME"])

        # 讀灰階影像（如需）
        rimg = cv2.imread(f"data/frames/{name}", cv2.IMREAD_GRAYSCALE)

        # 讀該影像的 2D 特徵與描述子（由 point_desc.pkl 提供）
        pts_df     = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query   = np.array(pts_df["XY"].to_list(), dtype=np.float32)          # (N,2)
        desc_query = np.array(pts_df["DESCRIPTORS"].to_list(), dtype=np.float32) # (N,128)

        ok, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        if not ok:
            continue

        # 轉四元數（xyzw）與平移
        R_est = R.from_rotvec(rvec.reshape(3))
        q_est_xyzw = R_est.as_quat()
        t_est = tvec.reshape(3)

        # 讀 GT：講義宣告 image.pkl 的順序是 (QW,QX,QY,QZ)，
        # 但 DataFrame 欄位是 ["QX","QY","QZ","QW"]，這樣擷取後正好是 [x,y,z,w] 可直接用。:contentReference[oaicite:6]{index=6}
        q_gt_xyzw = row[["QX","QY","QZ","QW"]].values.astype(np.float64)
        t_gt      = row[["TX","TY","TZ"]].values.astype(np.float64)

        # 誤差
        rot_err_deg  = rotation_error(q_est_xyzw, q_gt_xyzw)
        tran_err     = translation_error(t_est, t_gt)

        r_list.append(q_est_xyzw)
        t_list.append(t_est)
        rot_err_list.append(rot_err_deg)
        trans_err_list.append(tran_err)

    # 中位數誤差（Problem 2 Step 2 要求）:contentReference[oaicite:7]{index=7}
    if len(rot_err_list) > 0:
        print(f"Median rotation error (deg): {np.median(rot_err_list):.3f}")
        print(f"Median translation error:    {np.median(trans_err_list):.3f}")
    else:
        print("No successful poses estimated.")

    # 視覺化（Problem 2 Step 3）：把 W2C 轉成 C2W 畫四角錐與軌跡:contentReference[oaicite:8]{index=8}
    Camera2World_Transform_Matrixs = []
    for q_xyzw, t in zip(r_list, t_list):
        R_wc = R.from_quat(q_xyzw).as_matrix().T       # R_cw = R^T；這裡先取 R_wc = (R_w2c)^T
        C_w  = -R_wc @ t.reshape(3,1)                  # 相機中心：C = -R^T t
        T_wc = np.eye(4)
        T_wc[:3,:3] = R_wc
        T_wc[:3, 3] = C_w.reshape(3)
        Camera2World_Transform_Matrixs.append(T_wc)

    if len(Camera2World_Transform_Matrixs) > 0:
        visualization(Camera2World_Transform_Matrixs, points3D_df)
