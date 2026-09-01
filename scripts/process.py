from pathlib import Path
import shutil


def flatten_and_rename_objs(source_folder):
    source_path = Path(source_folder)

    if not source_path.exists():
        print(f"Folder does not exist: {source_folder}")
        return

    output_path = source_path / "flattened_objs"
    output_path.mkdir(exist_ok=True)
    obj_files = [f for f in source_path.rglob("*") if f.suffix.lower() == ".obj"]
    obj_files = [f for f in obj_files if output_path not in f.parents]

    if not obj_files:
        print("No .obj files found.")
        return

    print(f"Found {len(obj_files)} .obj files")

    for index, file_path in enumerate(obj_files, start=1):
        destination_file = output_path / f"{index}.obj"
        shutil.copy2(file_path, destination_file)
        print(f"[{index}] {file_path.relative_to(source_path)} -> {destination_file.name}")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    flatten_and_rename_objs(
        r"C:\Users\janss\Desktop\research\3d-weight-diffusion\Final-pipeline\data\raw_meshes\03001627"
    )