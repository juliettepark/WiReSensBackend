import numpy as np
import pandas as pd
import os
import json
from scipy.spatial.transform import Rotation as R

def normalize_to_wrist_frame(df, joint_keys, wrist_key):
    """ Transforms joint coordinates into a stabilized local wrist frame. """
    w_pos_cols = [f"{wrist_key}_Px", f"{wrist_key}_Py", f"{wrist_key}_Pz"]
    w_ori_cols = [f"{wrist_key}_Qx", f"{wrist_key}_Qy", f"{wrist_key}_Qz", f"{wrist_key}_Qw"]
    
    if not all(c in df.columns for c in w_pos_cols + w_ori_cols): return None

    wrist_pos = df[w_pos_cols].values
    wrist_oris = R.from_quat(df[w_ori_cols].values)
    
    local_joints = []
    for key in joint_keys:
        if key == wrist_key: continue
        j_pos_cols = [f"{key}_Px", f"{key}_Py", f"{key}_Pz"]
        rel_pos = df[j_pos_cols].values - wrist_pos
        local_pos = wrist_oris.inv().apply(rel_pos) 
        local_joints.append(local_pos)
        
    return np.hstack(local_joints)

def get_joint_cols(lbl, onlyX=False):
    if onlyX:
        return [f"{lbl}_Py"]
    else:
        return [f"{lbl}_Px", f"{lbl}_Py", f"{lbl}_Pz"]

"""
Class to convert raw bone values to a scalar displacement value for a specific interaction set.

Example usage:
converter = DisplacementConverter("Power sphere")
displacement = converter.get(bone_values) # Entire frame of raw bone values from Quest
"""
class DisplacementConverter:
    def __init__(self, interaction_set, wrist_key='R_XRHand_Wrist', normalize_wrist=True):
        """
        Initializes the converter by loading PCA artifacts and mapping out
        the interaction sets and DataFrame headers.
        """
        paths = {
            "Heavy Wrap Internal": "pca_artifacts/pose_displacement_heavywrap_internal/pca_eigenvectors.json",
            "Heavy Wrap External": "pca_artifacts/pose_displacement_heavy_external/pca_eigenvectors.json",
            "Non-prehensile": "pca_artifacts/pose_displacement_nonprehensile/pca_eigenvectors.json",
            "Adducted Thumb Internal": "pca_artifacts/pose_displacement_adducted_thumb/pca_eigenvectors.json",
            "Power sphere": "pca_artifacts/pose_displacement_power_sphere/pca_eigenvectors.json",
            "Lateral Pinch": "pca_artifacts/pose_displacement_lateral/pca_eigenvectors.json",
            "Precision sphere": "pca_artifacts/pose_displacement_precision_sphere/pca_eigenvectors.json",
            "Precision pinch": "pca_artifacts/pose_displacement_index_pinch/pca_eigenvectors.json"
        }
        self.interaction_set = interaction_set
        self.pca_artifacts_path = paths[interaction_set]
        self.wrist_key = wrist_key
        self.normalize_wrist = normalize_wrist
        
        # --- 1. Load Saved Model ---
        if not os.path.exists(self.pca_artifacts_path):
            raise FileNotFoundError(f"Artifact file not found at {self.pca_artifacts_path}")
            
        with open(self.pca_artifacts_path, 'r') as f:
            self.artifacts = json.load(f)

        # --- 2. Define Pose Joints Dictionary ---
        # You can expand this dictionary with all your sets.
        self.interaction_pose_joints = {
            "Power sphere": [
                'R_XRHand_ThumbTip', 'R_XRHand_IndexTip', 
                'R_XRHand_MiddleTip', 'R_XRHand_RingTip', 'R_XRHand_LittleTip'
            ],
            "Precision pinch": [
                'R_XRHand_ThumbTip', 'R_XRHand_IndexTip'
            ],
            "Heavy wrap internal": [
                'R_XRHand_ThumbTip', 'R_XRHand_IndexTip', 'R_XRHand_MiddleTip',
                'R_XRHand_RingTip', 'R_XRHand_LittleTip'
            ],
        }

        # --- 3. Pre-compute headers for fast DataFrame conversion ---
        self.bone_names = [
            "XRHand_Wrist", "XRHand_Palm",
            "XRHand_ThumbMetacarpal", "XRHand_ThumbProximal", "XRHand_ThumbDistal", "XRHand_ThumbTip",
            "XRHand_IndexMetacarpal", "XRHand_IndexProximal", "XRHand_IndexIntermediate", "XRHand_IndexDistal", "XRHand_IndexTip",
            "XRHand_MiddleMetacarpal", "XRHand_MiddleProximal", "XRHand_MiddleIntermediate", "XRHand_MiddleDistal", "XRHand_MiddleTip",
            "XRHand_RingMetacarpal", "XRHand_RingProximal", "XRHand_RingIntermediate", "XRHand_RingDistal", "XRHand_RingTip",
            "XRHand_LittleMetacarpal", "XRHand_LittleProximal", "XRHand_LittleIntermediate", "XRHand_LittleDistal", "XRHand_LittleTip"
        ]
        
        self.bone_headers = []
        for prefix in ["L", "R"]:
            for bone in self.bone_names:
                # 7 values per bone: Px, Py, Pz, Qx, Qy, Qz, Qw
                for suffix in ["Px", "Py", "Pz", "Qx", "Qy", "Qz", "Qw"]:
                    self.bone_headers.append(f"{prefix}_{bone}_{suffix}")

    def get(self, bone_values):
        """
        Takes a flat list of raw bone values and an interaction set key,
        returns the scalar magnitude of the principal component vector.
        """
        # --- 1. Validate Keys ---
        if self.interaction_set not in self.artifacts:
            print(f"Error: Key '{self.interaction_set}' not found in PCA JSON.")
            return 0.0
            
        if self.interaction_set not in self.interaction_pose_joints:
            print(f"Error: Pose joints not defined for '{self.interaction_set}' in class.")
            return 0.0

        # --- 2. Extract PCA Data ---
        model = self.artifacts[self.interaction_set]
        pose_vec = np.array(model['pose']['eigenvector'])
        pose_joints = self.interaction_pose_joints[self.interaction_set]

        # --- 3. Package into single-row DataFrame ---
        # (Assuming bone_values length matches len(self.bone_headers))
        df = pd.DataFrame([bone_values], columns=self.bone_headers)

        # --- 4. Normalize or Extract Coordinates ---
        if self.normalize_wrist:
            # Assuming normalize_to_wrist_frame is available in your global scope
            joints_local = normalize_to_wrist_frame(df, pose_joints, self.wrist_key)
        else:
            raw_data = []
            for j in pose_joints:
                # Assuming get_joint_cols is available in your global scope
                cols = get_joint_cols(j, onlyX=False)
                if all(c in df.columns for c in cols):
                    raw_data.append(df[cols].values)
            joints_local = np.hstack(raw_data) if raw_data else None

        # --- 5. Validate Dimensions ---
        if joints_local is None or joints_local.shape[1] != len(pose_vec):
            print(f"[Warning] Dimension mismatch. Expected {len(pose_vec)}, got {joints_local.shape[1] if joints_local is not None else 'None'}.")
            return 0.0

        # --- 6. Project and Return Scalar ---
        # joints_local is shape (1, N) and pose_vec is (N,) -> dot product yields 1D array of length 1
        pc1_mag = np.dot(joints_local, pose_vec)
        
        return float(pc1_mag[0]) if isinstance(pc1_mag, np.ndarray) else float(pc1_mag)