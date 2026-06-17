import torch
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.flow_utils import grasp_rot_to_ypr
from utils.loss_utils import batch_viewpoint_params_to_matrix

def main():
    print("Testing grasp_rot_to_ypr...")
    
    # 1. Create a simple approach vector (e.g., pointing straight along X)
    # In GraspNetAPI, axis_x is the approaching vector
    approach = torch.tensor([[1.0, 0.0, 0.0]]) 
    
    # 2. Create an in-plane angle (e.g., 90 degrees = pi/2)
    angle = torch.tensor([3.14159265 / 2.0])
    
    # 3. Generate the 3x3 rotation matrix using the project's built-in function
    matrix_3x3 = batch_viewpoint_params_to_matrix(approach, angle)
    print("--- Original 3x3 Matrix ---")
    print(matrix_3x3)
    
    # 4. Flatten to 9 values (simulating model output)
    matrix_9 = matrix_3x3.view(-1, 9)
    print("\n--- Flattened 9D Matrix ---")
    print(matrix_9)
    
    # 5. Test the grasp_rot_to_ypr function
    ypr_rad = grasp_rot_to_ypr(matrix_9, degrees=False)
    ypr_deg = grasp_rot_to_ypr(matrix_9, degrees=True)
    
    print("\n--- Result in Radians (Yaw, Pitch, Roll) ---")
    print(ypr_rad)
    
    print("\n--- Result in Degrees (Yaw, Pitch, Roll) ---")
    print(ypr_deg)

if __name__ == "__main__":
    main()
