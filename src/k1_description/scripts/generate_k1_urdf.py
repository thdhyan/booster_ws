#!/usr/bin/env python3
"""Generate a per-namespace K1 URDF for Gazebo.

Runs xacro on k1_gz.urdf.xacro (robot_ns parameterized) and rewrites the
relative mesh paths from booster_assets into absolute file:// URIs, since
the URDF is consumed as a string/file by `ros_gz_sim create`.

Usage:
  python3 generate_k1_urdf.py --robot_ns k1_0 [--out /tmp/k1_urdf/k1_0.urdf]
"""
import argparse
import os
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ns", default="k1_0")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from ament_index_python.packages import get_package_share_directory

    pkg = get_package_share_directory("k1_description")
    xacro_file = os.path.join(pkg, "urdf", "k1_gz.urdf.xacro")

    out_path = args.out or f"/tmp/k1_urdf/{args.robot_ns}.urdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    xml = subprocess.check_output(
        ["xacro", xacro_file, f"robot_ns:={args.robot_ns}"], text=True
    )

    # meshes/Trunk.STL -> file://<share>/assets/robots/K1/meshes/Trunk.STL
    asset_mesh_dir = os.path.join(
        pkg, "assets", "robots", "K1", "meshes"
    )
    xml = re.sub(r'filename="meshes/', f'filename="file://{asset_mesh_dir}/', xml)

    # The K1 URDF names some links identically to their joints (e.g. link
    # 'Left_Hip_Pitch' + joint 'Left_Hip_Pitch'). Legal in URDF, but the URDF->SDF
    # conversion for Gazebo requires model-wide unique names -> rename the LINKS
    # (suffix '_link'), keeping joint names canonical for control.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    joint_names = {j.get("name") for j in root.findall("joint")}
    renames = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name in joint_names:
            new = name + "_link"
            renames[name] = new
            link.set("name", new)
    if renames:
        for joint in root.findall("joint"):
            for tag in ("parent", "child"):
                ref = joint.find(tag)
                if ref is not None and ref.get("link") in renames:
                    ref.set("link", renames[ref.get("link")])
        xml = ET.tostring(root, encoding="unicode")

    with open(out_path, "w") as f:
        f.write(xml)
    print(out_path)


if __name__ == "__main__":
    sys.exit(main())
