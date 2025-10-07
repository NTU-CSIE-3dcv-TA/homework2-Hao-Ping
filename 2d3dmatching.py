from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
from tqdm import tqdm
import open3d as o3d

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

def pnpsolver(query, model, cameraMatrix=0, distortion=0):
    """
    1) 2D-3D 對應：query 的 2D keypoints/desc 與 model 的 3D desc 做 NN matching
    2) ratio test 過濾
    3) solvePnPRansac 求解 W2C 的 (R,t)
    備註：本作業講義採 p = K [R|T] X（W2C）:contentReference[oaicite:1]{index=1}
    """
    kp_query, desc_query = query            # kp_query: (N,2) 像素座標；desc_query: (N,128) float32
    kp_model, desc_model = model            # kp_model: (M,3) 世界座標；desc_model: (M,128) float32

    # 以講義/骨架預設內參與畸變（Brown-Conrady，順序 k1,k2,p1,p2）:contentReference[oaicite:2]{index=2}
    cameraMatrix = np.array([[1868.27, 0,   540],
                             [0,       1869.18, 960],
                             [0,       0,       1  ]], dtype=np.float64)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float64)

    if len(desc_query) == 0 or len(desc_model) == 0:
        return False, None, None, None

    matcher = _build_matcher()
    # knn=2 for ratio test
    matches = matcher.knnMatch(desc_query, desc_model, k=2)

    good_pts2d = []
    good_pts3d = []
    ratio = 0.8
    for m, n in matches:
        if m.distance < ratio * n.distance:
            q_idx = m.queryIdx
            m_idx = m.trainIdx
            good_pts2d.append(kp_query[q_idx])
            good_pts3d.append(kp_model[m_idx])

    if len(good_pts2d) < 6:
        # 點太少，不足以穩定 PnP
        return False, None, None, None

    pts2d = np.asarray(good_pts2d, dtype=np.float64)
    pts3d = np.asarray(good_pts3d, dtype=np.float64)

    # RANSAC pnp（輸出 rvec,tvec 對應 W2C）
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
    """
    畫：3D 點雲 + 相機軌跡（四角錐）
    Problem 2 Step 3 要求用 Open3D 顯示相機與點雲並討論。:contentReference[oaicite:4]{index=4}
    """
    geoms = []

    # 點雲
    if "XYZ" in points3D_df.columns:
        pts = np.vstack(points3D_df["XYZ"].to_list())
    else:
        # 若 points3D_df 已經展開 XYZx,XYZy,XYZz
        pts = points3D_df[["X","Y","Z"]].values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    geoms.append(pcd)

    # 相機四角錐 + 軌跡
    traj_xyz = []
    for T in Camera2World_Transform_Matrixs:
        geoms.append(_make_frustum_lineset(T, scale=0.6))
        cam_center = T[:3, 3]
        traj_xyz.append(cam_center)
    if len(traj_xyz) >= 2:
        traj = o3d.geometry.LineSet()
        traj_pts = np.array(traj_xyz, dtype=np.float64)
        traj.points = o3d.utility.Vector3dVector(traj_pts)
        traj_lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        traj.lines  = o3d.utility.Vector2iVector(traj_lines)
        geoms.append(traj)

    o3d.visualization.draw_geometries(geoms)

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
    valid_df   = images_df[valid_mask].sort_values("IMAGE_ID")

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
