import trimesh
from stl import mesh
import mujoco
import os

class Helpers:
    @staticmethod
    def urdf_to_xml(): # Requires reformatting using <mujocoinclude> tag
        base_path = os.path.join(os.getcwd(), 'Simulation', 'Meshes')
        model = mujoco.MjModel.from_xml_path(os.path.join(base_path, 'robot.urdf'))
        mujoco.mj_saveLastXML(os.path.join(base_path, 'robot.xml'), model)
        
    @staticmethod
    def face_reduce():
        for file in os.listdir(os.path.join(os.getcwd(), 'Simulation', 'Meshes')):
            if file.lower().endswith('stl'):
                path = os.path.join(os.getcwd(), 'Simulation', 'Meshes', file)
                mesh = trimesh.load(path)
                if len(mesh.faces) > 3000:
                    simplified_mesh = mesh.simplify_quadric_decimation(face_count=3000, aggression=4)
                    simplified_mesh.export(path)
                    print(f"Reduced {len(mesh.faces)} to {len(simplified_mesh.faces)} faces for part {file}")

if __name__ == "__main__":
    Helpers.face_reduce()