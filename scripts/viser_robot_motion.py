"""Interactive browser viewer (viser) for retargeted robot motion.

Works headless: open the printed URL (default http://localhost:8080) in a browser.

Examples:
    # play a saved motion
    python scripts/viser_robot_motion.py --robot sprout --robot_motion_path sprout_fight.pkl
    # retarget a BVH on the fly, then play it with the human skeleton overlaid
    python scripts/viser_robot_motion.py --robot sprout --bvh_file ~/motion_data/LAFAN1/dance1_subject2.bvh
"""
import argparse
import os
import time

import mujoco as mj
import numpy as np
import viser
from tqdm import tqdm

from general_motion_retargeting import ROBOT_XML_DICT, load_robot_motion

DEFAULT_RGBA = np.array([0.85, 0.35, 0.15, 1.0])


def visual_mesh_geoms(model):
    """Mesh geoms used for rendering: non-colliding ones if the model has any, else all meshes."""
    meshes = [g for g in range(model.ngeom) if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
    visual = [g for g in meshes if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return visual or meshes


def geom_rgba(model, g):
    mat = model.geom_matid[g]
    if mat >= 0:
        # textured materials have no meaningful flat colour in viser
        if model.mat_texid[mat].max() >= 0:
            return DEFAULT_RGBA
        return model.mat_rgba[mat]
    return model.geom_rgba[g]


def add_robot(server, model):
    handles = {}
    for g in visual_mesh_geoms(model):
        mesh_id = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        rgba = geom_rgba(model, g)
        handles[g] = server.scene.add_mesh_simple(
            f"/robot/geom_{g}",
            vertices=model.mesh_vert[v0:v0 + nv].astype(np.float32),
            faces=model.mesh_face[f0:f0 + nf].astype(np.uint32),
            color=tuple(int(c * 255) for c in rgba[:3]),
            opacity=float(rgba[3]) if rgba[3] < 1.0 else None,
        )
    return handles


def update_robot(model, data, handles, qpos):
    data.qpos[:] = qpos
    mj.mj_kinematics(model, data)
    quat = np.zeros(4)
    for g, h in handles.items():
        mj.mju_mat2Quat(quat, data.geom_xmat[g])
        h.position = data.geom_xpos[g].copy()
        h.wxyz = quat.copy()


def retarget_bvh(bvh_file, robot, fmt):
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(bvh_file, format=fmt)
    retargeter = GeneralMotionRetargeting(
        src_human=f"bvh_{fmt}", tgt_robot=robot, actual_human_height=human_height, verbose=False
    )
    qpos_list, human_list = [], []
    for f in tqdm(frames, desc="Retargeting"):
        qpos_list.append(retargeter.retarget(f))
        human_list.append(
            np.stack([retargeter.scaled_human_data[b][0] for b in retargeter.human_body_to_task2])
        )
    return np.array(qpos_list), np.array(human_list)


def load_motion(path):
    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(path)
    return fps, np.concatenate([root_pos, root_rot, dof_pos], axis=1)


def save_motion(path, qpos, fps):
    import pickle

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "fps": fps,
            "root_pos": qpos[:, :3],
            "root_rot": qpos[:, [4, 5, 6, 3]],  # wxyz -> xyzw
            "dof_pos": qpos[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }, f)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1", choices=sorted(ROBOT_XML_DICT))
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--robot_motion_path", help="Saved robot motion (.pkl)")
    src.add_argument("--bvh_file", help="BVH file to retarget before playing")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=int, default=30, help="FPS of the BVH input")
    parser.add_argument("--save_path", default=None, help="Also save the retargeted BVH motion")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    human = None
    if args.bvh_file:
        fps = args.motion_fps
        qpos, human = retarget_bvh(os.path.expanduser(args.bvh_file), args.robot, args.format)
        if args.save_path:
            save_motion(args.save_path, qpos, fps)
    else:
        fps, qpos = load_motion(args.robot_motion_path)
    num_frames = len(qpos)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=20.0, height=20.0, cell_size=0.5)
    handles = add_robot(server, model)

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        frame = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
    with server.gui.add_folder("Display"):
        follow = server.gui.add_checkbox("Camera follows robot", initial_value=False)
        show_human = server.gui.add_checkbox("Show human targets", initial_value=human is not None,
                                             disabled=human is None)
    server.gui.add_markdown(f"**{args.robot}** · {num_frames} frames @ {fps} fps")

    human_points = None
    if human is not None:
        human_points = server.scene.add_point_cloud(
            "/human", points=human[0], colors=(30, 144, 255), point_size=0.03, point_shape="circle"
        )

    shown = -1
    t_last = time.time()
    cursor = 0.0
    print(f"Viser running: open http://localhost:{args.port} in a browser (Ctrl+C to quit)")
    while True:
        now = time.time()
        dt, t_last = now - t_last, now
        if playing.value:
            cursor += dt * fps * speed.value
            if cursor >= num_frames:
                if loop.value:
                    cursor %= num_frames
                else:
                    cursor, playing.value = num_frames - 1, False
            if int(cursor) != frame.value:
                frame.value = int(cursor)
        else:
            cursor = float(frame.value)

        i = frame.value
        if i != shown:
            with server.atomic():
                update_robot(model, data, handles, qpos[i])
                if human_points is not None:
                    human_points.points = human[i]
            shown = i
            if follow.value:
                root = qpos[i, :3]
                for client in server.get_clients().values():
                    client.camera.look_at = root
        if human_points is not None:
            human_points.visible = show_human.value
        time.sleep(1.0 / 120)


if __name__ == "__main__":
    main()
