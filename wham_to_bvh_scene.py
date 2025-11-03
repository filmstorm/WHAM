#!/usr/bin/env python3
"""
WHAM to Unified Scene BVH Converter
Exports all detected people to a SINGLE unified BVH file with all characters in one scene.
Each character becomes a separate branch in the hierarchy, all spatially aligned.
"""
import argparse
import numpy as np
import joblib
import os
import json
from scipy.spatial.transform import Rotation as R

# SMPL skeleton bone mapping (24 joints)
part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand',
              'bone_23': 'R_Hand'}

# SMPL joint offsets (in cm, for BVH visualization)
JOINT_OFFSETS = {
    'root': [0.0, 0.0, 0.0],
    'Pelvis': [0.0, 95.0, 0.0],
    'L_Hip': [9.0, 0.0, 0.0],
    'R_Hip': [-9.0, 0.0, 0.0],
    'Spine1': [0.0, 7.0, 0.0],
    'L_Knee': [0.0, -45.0, 0.0],
    'R_Knee': [0.0, -45.0, 0.0],
    'Spine2': [0.0, 10.0, 0.0],
    'L_Ankle': [0.0, -45.0, 0.0],
    'R_Ankle': [0.0, -45.0, 0.0],
    'Spine3': [0.0, 10.0, 0.0],
    'L_Foot': [0.0, -10.0, 15.0],
    'R_Foot': [0.0, -10.0, 15.0],
    'Neck': [0.0, 10.0, 0.0],
    'L_Collar': [7.0, 3.0, 0.0],
    'R_Collar': [-7.0, 3.0, 0.0],
    'Head': [0.0, 10.0, 0.0],
    'L_Shoulder': [10.0, 0.0, 0.0],
    'R_Shoulder': [-10.0, 0.0, 0.0],
    'L_Elbow': [30.0, 0.0, 0.0],
    'R_Elbow': [-30.0, 0.0, 0.0],
    'L_Wrist': [25.0, 0.0, 0.0],
    'R_Wrist': [-25.0, 0.0, 0.0],
    'L_Hand': [8.0, 0.0, 0.0],
    'R_Hand': [-8.0, 0.0, 0.0],
}

def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle representation to rotation matrix"""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-6:
        return np.eye(3)

    axis = axis_angle / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

def pose_to_rotation_matrices(pose):
    """Convert SMPL pose parameters to rotation matrices"""
    pose = pose.reshape(24, 3)
    rotation_matrices = []
    for i in range(24):
        rot_mat = axis_angle_to_rotation_matrix(pose[i])
        rotation_matrices.append(rot_mat)
    return rotation_matrices

def convert_transl(transl):
    """Convert translation from meters to centimeters for BVH visibility"""
    return transl * 100.0

def matrix_to_euler(mat):
    """Convert rotation matrix to Euler angles (in degrees) in ZXY order for BVH"""
    r = R.from_matrix(mat)
    return r.as_euler('ZXY', degrees=True)

def create_character_hierarchy(char_name, depth=0):
    """Create a complete character skeleton hierarchy"""
    joint_hierarchy = {
        'root': ['Pelvis'],
        'Pelvis': ['L_Hip', 'R_Hip', 'Spine1'],
        'L_Hip': ['L_Knee'],
        'R_Hip': ['R_Knee'],
        'Spine1': ['Spine2'],
        'L_Knee': ['L_Ankle'],
        'R_Knee': ['R_Ankle'],
        'Spine2': ['Spine3'],
        'L_Ankle': ['L_Foot'],
        'R_Ankle': ['R_Foot'],
        'Spine3': ['Neck', 'L_Collar', 'R_Collar'],
        'Neck': ['Head'],
        'L_Collar': ['L_Shoulder'],
        'R_Collar': ['R_Shoulder'],
        'L_Shoulder': ['L_Elbow'],
        'R_Shoulder': ['R_Elbow'],
        'L_Elbow': ['L_Wrist'],
        'R_Elbow': ['R_Wrist'],
        'L_Wrist': ['L_Hand'],
        'R_Wrist': ['R_Hand'],
        'Head': [],
        'L_Foot': [],
        'R_Foot': [],
        'L_Hand': [],
        'R_Hand': []
    }

    def create_joint(joint_name, d=0):
        """Recursively create joint definitions"""
        indent = "  " * (depth + d)
        joint_full_name = f"{char_name}_{joint_name}"
        offset = JOINT_OFFSETS[joint_name]

        if joint_name == 'root':
            # Character root has position + rotation channels
            result = f"{indent}JOINT {char_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            result += f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        else:
            result = f"{indent}JOINT {joint_full_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            result += f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n"

        children = joint_hierarchy[joint_name]
        for child in children:
            result += create_joint(child, d + 1)

        if not children:  # End Site
            result += f"{indent}  End Site\n"
            result += f"{indent}  {{\n"
            if joint_name in ['L_Hand', 'R_Hand']:
                result += f"{indent}    OFFSET {5.0:.6f} {0.0:.6f} {0.0:.6f}\n"
            elif joint_name in ['L_Foot', 'R_Foot']:
                result += f"{indent}    OFFSET {0.0:.6f} {0.0:.6f} {10.0:.6f}\n"
            elif joint_name == 'Head':
                result += f"{indent}    OFFSET {0.0:.6f} {10.0:.6f} {0.0:.6f}\n"
            else:
                result += f"{indent}    OFFSET {0.0:.6f} {5.0:.6f} {0.0:.6f}\n"
            result += f"{indent}  }}\n"

        result += f"{indent}}}\n"
        return result

    return create_joint('root')

def create_scene_hierarchy(character_names):
    """Create unified scene hierarchy with all characters"""
    hierarchy = "HIERARCHY\n"
    hierarchy += "ROOT Scene\n"
    hierarchy += "{\n"
    hierarchy += "  OFFSET 0.000000 0.000000 0.000000\n"
    hierarchy += "  CHANNELS 0\n"  # Scene root has no channels

    # Add each character as a child of Scene
    for char_name in character_names:
        hierarchy += create_character_hierarchy(char_name, depth=1)

    hierarchy += "}\n"
    return hierarchy

def create_character_joint_order(char_name):
    """Create joint order for one character"""
    joint_order = [char_name]  # Character root

    joint_hierarchy = {
        'root': ['Pelvis'],
        'Pelvis': ['L_Hip', 'R_Hip', 'Spine1'],
        'L_Hip': ['L_Knee'],
        'R_Hip': ['R_Knee'],
        'Spine1': ['Spine2'],
        'L_Knee': ['L_Ankle'],
        'R_Knee': ['R_Ankle'],
        'Spine2': ['Spine3'],
        'L_Ankle': ['L_Foot'],
        'R_Ankle': ['R_Foot'],
        'Spine3': ['Neck', 'L_Collar', 'R_Collar'],
        'Neck': ['Head'],
        'L_Collar': ['L_Shoulder'],
        'R_Collar': ['R_Shoulder'],
        'L_Shoulder': ['L_Elbow'],
        'R_Shoulder': ['R_Elbow'],
        'L_Elbow': ['L_Wrist'],
        'R_Elbow': ['R_Wrist'],
        'L_Wrist': ['L_Hand'],
        'R_Wrist': ['R_Hand'],
        'Head': [],
        'L_Foot': [],
        'R_Foot': [],
        'L_Hand': [],
        'R_Hand': []
    }

    def traverse(joint_name):
        children = joint_hierarchy[joint_name]
        for child in children:
            joint_order.append(f"{char_name}_{child}")
            traverse(child)

    traverse('root')
    return joint_order

def create_joint_to_bone_mapping():
    """Create mapping from joint names to bone indices"""
    joint_to_bone = {}
    for bone_id, joint_name in part_match.items():
        if bone_id != 'root':
            bone_index = int(bone_id.split('_')[1])
            joint_to_bone[joint_name] = bone_index
    return joint_to_bone

def create_t_pose_character(char_joint_order):
    """Create T-pose frame data for one character"""
    frame_data = []
    for joint in char_joint_order:
        if joint == char_joint_order[0]:  # Character root
            # Position at origin
            frame_data.extend([0.0, 0.0, 0.0])
            # No rotation
            frame_data.extend([0.0, 0.0, 0.0])
        else:
            # Neutral rotation for all joints
            frame_data.extend([0.0, 0.0, 0.0])
    return frame_data

def wham_scene_to_bvh(input_file, output_file, fps=30, min_frames=30):
    """
    Convert WHAM output to unified scene BVH with all characters in one file

    Args:
        input_file: Path to WHAM wham_output.pkl file
        output_file: Path to output unified BVH file
        fps: Frames per second
        min_frames: Minimum frames required to include a person
    """
    print(f"Converting WHAM multi-person output to unified scene BVH")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load WHAM results
    results = joblib.load(input_file)
    print(f"\nDetected {len(results)} people in scene")

    # Filter subjects by minimum frame count
    valid_subjects = {}
    for subject_id, data in results.items():
        n_frames = len(data['frame_ids'])
        if n_frames >= min_frames:
            valid_subjects[subject_id] = data
            print(f"  Character_{subject_id:02d}: {n_frames} frames ✓")
        else:
            print(f"  Subject {subject_id}: {n_frames} frames (skipped - below minimum)")

    if not valid_subjects:
        print("No subjects meet minimum frame requirement!")
        return

    # Determine scene-wide frame range
    all_frame_ids = []
    for data in valid_subjects.values():
        all_frame_ids.extend(data['frame_ids'])
    scene_start_frame = min(all_frame_ids)
    scene_end_frame = max(all_frame_ids)
    scene_total_frames = scene_end_frame - scene_start_frame + 1

    print(f"\nScene frame range: {scene_start_frame} to {scene_end_frame} ({scene_total_frames} frames)")

    # Create character names
    character_names = [f"Character_{sid:02d}" for sid in valid_subjects.keys()]

    # Create hierarchy and joint orders
    hierarchy = create_scene_hierarchy(character_names)
    joint_to_bone = create_joint_to_bone_mapping()

    # Create joint order for each character
    all_joint_orders = {}
    for sid, char_name in zip(valid_subjects.keys(), character_names):
        all_joint_orders[sid] = create_character_joint_order(char_name)

    # Reference Y for grounding
    first_subject = list(valid_subjects.values())[0]
    first_y = convert_transl(first_subject['trans_world'][0])[1]

    # Prepare frame data for all characters
    character_data = {}
    for subject_id, subject_data in valid_subjects.items():
        pose_world = subject_data['pose_world']
        trans_world = subject_data['trans_world']
        frame_ids = subject_data['frame_ids']
        person_start_offset = frame_ids[0] - scene_start_frame

        character_data[subject_id] = {
            'pose_world': pose_world,
            'trans_world': trans_world,
            'frame_ids': frame_ids,
            'start_offset': person_start_offset
        }

    # Write unified BVH file
    frame_time = 1.0 / fps

    with open(output_file, 'w') as f:
        # Write hierarchy
        f.write(hierarchy)

        # Write motion section
        f.write("MOTION\n")
        f.write(f"Frames: {scene_total_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        # Process each frame
        for scene_frame_idx in range(scene_total_frames):
            frame_data = []

            # Process each character
            for subject_id in valid_subjects.keys():
                char_info = character_data[subject_id]
                person_frame_idx = scene_frame_idx - char_info['start_offset']

                if 0 <= person_frame_idx < len(char_info['frame_ids']):
                    # Character is present - use actual data
                    trans = convert_transl(char_info['trans_world'][person_frame_idx])
                    trans[1] = trans[1] - first_y  # Ground relative to scene
                    pose = char_info['pose_world'][person_frame_idx]
                    mrots = pose_to_rotation_matrices(pose)
                    root_rotation = mrots[0]

                    # Write character data
                    joint_order = all_joint_orders[subject_id]
                    for joint in joint_order:
                        if joint == joint_order[0]:  # Character root
                            # Root position
                            frame_data.extend(trans)
                            # Root has no rotation (applied to pelvis)
                            frame_data.extend([0.0, 0.0, 0.0])
                        elif joint.endswith('_Pelvis'):
                            # Apply root rotation to pelvis
                            euler = matrix_to_euler(root_rotation)
                            frame_data.extend(euler)
                        else:
                            # Find bone index from joint name
                            joint_base = joint.split('_', 1)[1]  # Remove character prefix
                            bone_index = joint_to_bone.get(joint_base)

                            if bone_index is not None:
                                euler = matrix_to_euler(mrots[bone_index])
                                frame_data.extend(euler)
                            else:
                                frame_data.extend([0.0, 0.0, 0.0])
                else:
                    # Character not present - T-pose at origin
                    t_pose = create_t_pose_character(all_joint_orders[subject_id])
                    frame_data.extend(t_pose)

            # Write frame
            f.write(' '.join([f"{x:.6f}" for x in frame_data]) + '\n')

    print(f"\n✓ Unified scene BVH created successfully!")
    print(f"  - {len(valid_subjects)} characters in one file")
    print(f"  - {scene_total_frames} frames at {fps} FPS")
    print(f"  - All characters in unified world-space")
    print(f"\nImport {output_file} to see the complete scene with all characters!")

def main():
    parser = argparse.ArgumentParser(
        description='Convert WHAM multi-person output to unified scene BVH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create unified scene BVH with all people
  python wham_to_bvh_scene.py -i output/demo/video/wham_output.pkl -o scene.bvh

  # Custom minimum frames threshold
  python wham_to_bvh_scene.py -i wham_output.pkl -o scene.bvh --min-frames 50

  # Custom FPS
  python wham_to_bvh_scene.py -i wham_output.pkl -o scene.bvh --fps 60
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input WHAM pickle file (wham_output.pkl)')
    parser.add_argument('--output', '-o', required=True, help='Output unified scene BVH file')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Frames per second (default: 30)')
    parser.add_argument('--min-frames', type=int, default=30, help='Minimum frames required to include a person (default: 30)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    wham_scene_to_bvh(args.input, args.output, args.fps, args.min_frames)

if __name__ == '__main__':
    main()
