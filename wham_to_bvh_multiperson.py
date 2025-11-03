#!/usr/bin/env python3
"""
WHAM to Multi-Person BVH Converter
Exports all detected people to synchronized BVH files in a unified world-space scene.
Each person gets their own BVH file, all spatially aligned and frame-synchronized.
"""
import argparse
import numpy as np
import joblib
import os
import json
import math
from scipy.spatial.transform import Rotation as R

# Import the single-person converter functions
import sys
sys.path.insert(0, os.path.dirname(__file__))

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

    # Rodrigues formula
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

def rotate180(rot):
    """WHAM already uses Y-up right-handed coordinates matching BVH"""
    return rot

def convert_transl(transl):
    """Convert translation from meters to centimeters for BVH visibility"""
    return transl * 100.0

def matrix_to_euler(mat):
    """Convert rotation matrix to Euler angles (in degrees) in ZXY order for BVH"""
    r = R.from_matrix(mat)
    return r.as_euler('ZXY', degrees=True)

def create_hierarchy():
    """Create the BVH hierarchy section with proper bone offsets"""
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

    def create_joint(joint_name, depth=0):
        """Recursively create joint definitions with proper indentation"""
        indent = "  " * depth
        offset = JOINT_OFFSETS[joint_name]

        if joint_name == 'root':
            result = f"{indent}ROOT {joint_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            result += f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        else:
            result = f"{indent}JOINT {joint_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            result += f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n"

        children = joint_hierarchy[joint_name]
        for child in children:
            result += create_joint(child, depth + 1)

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

    hierarchy = "HIERARCHY\n"
    hierarchy += create_joint('root')
    return hierarchy

def create_joint_order():
    """Create a flat list of joints in the order they appear in the hierarchy"""
    joint_order = ['root']

    def traverse_hierarchy(joint_name):
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

        children = joint_hierarchy[joint_name]
        for child in children:
            joint_order.append(child)
            traverse_hierarchy(child)

    traverse_hierarchy('root')
    return joint_order

def create_joint_to_bone_mapping():
    """Create a mapping from joint names to bone indices in the SMPL model"""
    joint_to_bone = {}
    for bone_id, joint_name in part_match.items():
        if bone_id != 'root':
            bone_index = int(bone_id.split('_')[1])
            joint_to_bone[joint_name] = bone_index
    return joint_to_bone

def create_t_pose_frame(joint_order, joint_to_bone):
    """Create a T-pose frame (neutral position at origin)"""
    frame_data = []
    for joint in joint_order:
        if joint == 'root':
            # Root at origin
            frame_data.extend([0.0, 0.0, 0.0])
            frame_data.extend([0.0, 0.0, 0.0])
        else:
            # Neutral rotation
            frame_data.extend([0.0, 0.0, 0.0])
    return frame_data

def wham_multiperson_to_bvh(input_file, output_dir, fps=30, min_frames=30, sync_frames=True):
    """
    Convert WHAM pickle file with multiple people to synchronized BVH files

    Args:
        input_file: Path to WHAM wham_output.pkl file
        output_dir: Directory to save BVH files and metadata
        fps: Frames per second
        min_frames: Minimum frames required to export a person
        sync_frames: If True, synchronize all BVH files to same total frame count
    """
    print(f"Converting WHAM multi-person output: {input_file}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load WHAM results
    results = joblib.load(input_file)

    print(f"\nDetected {len(results)} people in scene")

    # Filter subjects by minimum frame count
    valid_subjects = {}
    for subject_id, data in results.items():
        n_frames = len(data['frame_ids'])
        if n_frames >= min_frames:
            valid_subjects[subject_id] = data
            print(f"  Subject {subject_id}: {n_frames} frames ✓")
        else:
            print(f"  Subject {subject_id}: {n_frames} frames (skipped - below minimum)")

    if not valid_subjects:
        print("No subjects meet minimum frame requirement!")
        return

    # Determine scene-wide frame range
    if sync_frames:
        all_frame_ids = []
        for data in valid_subjects.values():
            all_frame_ids.extend(data['frame_ids'])
        scene_start_frame = min(all_frame_ids)
        scene_end_frame = max(all_frame_ids)
        scene_total_frames = scene_end_frame - scene_start_frame + 1
        print(f"\nScene frame range: {scene_start_frame} to {scene_end_frame} ({scene_total_frames} frames)")
    else:
        scene_start_frame = 0
        scene_end_frame = 0
        scene_total_frames = 0

    # Get the frame time
    frame_time = 1.0 / fps

    # Create hierarchy and joint info (same for all people)
    hierarchy = create_hierarchy()
    joint_order = create_joint_order()
    joint_to_bone = create_joint_to_bone_mapping()

    # Scene metadata (convert all to native Python types for JSON)
    scene_metadata = {
        'fps': float(fps),
        'scene_frames': int(scene_total_frames) if sync_frames else None,
        'frame_range': [int(scene_start_frame), int(scene_end_frame)] if sync_frames else None,
        'sync_frames': bool(sync_frames),
        'people': []
    }

    # Reference Y for grounding (use first valid subject)
    first_subject = list(valid_subjects.values())[0]
    first_y = convert_transl(first_subject['trans_world'][0])[1]

    # Export each person
    for subject_id, subject_data in valid_subjects.items():
        output_file = os.path.join(output_dir, f"person_{subject_id:02d}.bvh")

        pose_world = subject_data['pose_world']
        trans_world = subject_data['trans_world']
        frame_ids = subject_data['frame_ids']

        # Calculate position stats
        trans_cm = convert_transl(trans_world)
        center_pos = trans_cm.mean(axis=0)

        print(f"\nExporting Subject {subject_id}:")
        print(f"  Frames: {len(frame_ids)}")
        print(f"  Center position: X={center_pos[0]:.1f} Y={center_pos[1]:.1f} Z={center_pos[2]:.1f} cm")

        # Determine frame count for this person's BVH
        if sync_frames:
            # Pad to match scene frame count
            nFrames = scene_total_frames
            person_start_offset = frame_ids[0] - scene_start_frame
        else:
            nFrames = len(frame_ids)
            person_start_offset = 0

        # Open the file for writing
        with open(output_file, 'w') as f:
            # Write the HIERARCHY section
            f.write(hierarchy)

            # Write the MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {nFrames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            # Process each frame
            for scene_frame_idx in range(nFrames):
                if sync_frames:
                    # Check if this person is present in this frame
                    person_frame_idx = scene_frame_idx - person_start_offset

                    if 0 <= person_frame_idx < len(frame_ids):
                        # Person is present - use actual data
                        trans = convert_transl(trans_world[person_frame_idx])
                        trans[1] = trans[1] - first_y  # Ground relative to scene
                        pose = pose_world[person_frame_idx]
                        mrots = pose_to_rotation_matrices(pose)
                        root_rotation = rotate180(mrots[0])
                    else:
                        # Person not present - use T-pose at origin (invisible)
                        frame_data = create_t_pose_frame(joint_order, joint_to_bone)
                        f.write(' '.join([f"{x:.6f}" for x in frame_data]) + '\n')
                        continue
                else:
                    # No sync - just use sequential frames
                    trans = convert_transl(trans_world[scene_frame_idx])
                    trans[1] = trans[1] - first_y  # Ground relative to scene
                    pose = pose_world[scene_frame_idx]
                    mrots = pose_to_rotation_matrices(pose)
                    root_rotation = rotate180(mrots[0])

                # Convert rotation matrices to Euler angles and write motion data
                frame_data = []

                # Process each joint in order
                for joint in joint_order:
                    if joint == 'root':
                        # Root position
                        frame_data.extend(trans)
                        # Root has no rotation (applied to pelvis)
                        frame_data.extend([0.0, 0.0, 0.0])
                    elif joint == 'Pelvis':
                        # Apply the root rotation to pelvis
                        euler = matrix_to_euler(root_rotation)
                        frame_data.extend(euler)
                    else:
                        # Find the bone index for this joint
                        bone_index = joint_to_bone.get(joint)

                        if bone_index is not None:
                            # Add joint rotation
                            euler = matrix_to_euler(mrots[bone_index])
                            frame_data.extend(euler)
                        else:
                            # Joint not found in SMPL model, add default rotation
                            frame_data.extend([0.0, 0.0, 0.0])

                # Write frame data as a space-separated line
                f.write(' '.join([f"{x:.6f}" for x in frame_data]) + '\n')

        print(f"  ✓ Saved: {output_file}")

        # Add to metadata (convert numpy types to native Python types)
        scene_metadata['people'].append({
            'subject_id': int(subject_id),
            'bvh_file': f"person_{subject_id:02d}.bvh",
            'frames': int(len(frame_ids)),
            'frame_range': [int(frame_ids[0]), int(frame_ids[-1])],
            'center_position_cm': [float(center_pos[0]), float(center_pos[1]), float(center_pos[2])]
        })

    # Save scene metadata
    metadata_file = os.path.join(output_dir, 'scene_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scene_metadata, f, indent=2)

    print(f"\n✓ Multi-person scene exported successfully!")
    print(f"  - {len(valid_subjects)} people exported")
    print(f"  - Scene metadata: {metadata_file}")
    print(f"  - All characters in unified world-space coordinate system")
    if sync_frames:
        print(f"  - Synchronized to {scene_total_frames} frames at {fps} FPS")
    print(f"\nImport all BVH files together to see the complete scene!")

def main():
    parser = argparse.ArgumentParser(
        description='Convert WHAM multi-person output to synchronized BVH files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all people with 30+ frames, synchronized
  python wham_to_bvh_multiperson.py -i output/demo/video/wham_output.pkl -o output/demo/video/scene

  # Export with custom minimum frames
  python wham_to_bvh_multiperson.py -i wham_output.pkl -o scene --min-frames 50

  # Export without frame synchronization (each person has their own timeline)
  python wham_to_bvh_multiperson.py -i wham_output.pkl -o scene --no-sync
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input WHAM pickle file (wham_output.pkl)')
    parser.add_argument('--output', '-o', required=True, help='Output directory for BVH files and metadata')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Frames per second (default: 30)')
    parser.add_argument('--min-frames', type=int, default=30, help='Minimum frames required to export a person (default: 30)')
    parser.add_argument('--no-sync', action='store_true', help='Disable frame synchronization (each person has own timeline)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("\nTo generate WHAM output, run demo.py with --save_pkl flag:")
        print("  python demo.py --video VIDEO --save_pkl --visualize")
        return

    wham_multiperson_to_bvh(
        args.input,
        args.output,
        args.fps,
        args.min_frames,
        sync_frames=not args.no_sync
    )

if __name__ == '__main__':
    main()
