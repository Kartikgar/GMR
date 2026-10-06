# Sprout Models

This archive contains the Sprout robot models we use internally, offered in several
formats so you can load Sprout into the simulator or renderer of your choice. Every
model shares one visual mesh between each rotational joint and at least one collision
mesh, with extra collision meshes on the grippers, feet, and lower legs to support
training for ground contact and object manipulation. All models are in meters.

## Simulation & robot description formats

- **`sprout.urdf`** — Unified Robot Description Format. The kinematic and dynamic
  description used by ROS and most robotics tooling; references the meshes under
  `meshes/`.
- **`sprout.usd`** — Universal Scene Description. Compatible with NVIDIA Isaac and
  supports reinforcement-learning training (for example, in Isaac Lab). Textures and
  meshes are packaged into the USD.
- **`sprout.xml`** — MuJoCo (MJCF) model of the robot on its own.
- **`scene.xml`** — MuJoCo scene that includes `sprout.xml` and adds a ground plane
  and lighting; load this to view or simulate Sprout in MuJoCo.

## Visualization format

- **`sprout.fbx`** — a low-poly version of Sprout for real-time visualization — in
  simulation, or rendered on the web, mobile, and edge devices — with under 3 MB of
  geometry and textures combined. It is a triangle mesh bound to a simple
  forward-kinematics (FK) skeleton, exported from Maya (Y-up, right-handed) with no
  animation controls or rigging; the skeleton's joint frames are derived from the
  revolute joints in `sprout.urdf`.

## Supporting assets

- **`meshes/visual/`** — the visual `.obj` meshes (with `.mtl` materials) referenced
  by `sprout.urdf`.
- **`meshes/collision/`** — the collision `.obj` meshes, plus the `.stl` shoe
  collision hulls for the feet.
- **`textures/robot_D.png`** — the diffuse (color) texture for the `sprout.fbx` mesh.
- **`textures/robot_N.png`** — the normal map for the `sprout.fbx` mesh, adding surface
  detail to the low-poly geometry. Apply both textures to the mesh's material. If the
  surface looks inverted once the normal map is applied, flip its green channel.
