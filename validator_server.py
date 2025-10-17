"""Simple server for chunk validation UI."""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIR = Path(__file__).parent / "data"


@app.route("/")
def index():
    return send_from_directory(".", "chunk_validator.html")


@app.route("/api/folders", methods=["GET"])
def list_folders():
    """List all version folders (v1, v2, etc.) with labeled subdirs."""
    folders = []
    for version_dir in sorted(DATA_DIR.glob("v*")):
        labeled_dir = version_dir / "labeled"
        if labeled_dir.exists() and labeled_dir.is_dir():
            folders.append(f"{version_dir.name}/labeled")
    return jsonify({"folders": folders})


@app.route("/api/files", methods=["GET"])
def list_files():
    """List all .txt files in a specific folder."""
    folder = request.args.get("folder", "v1/labeled")
    folder_path = DATA_DIR / folder

    if not folder_path.exists():
        return jsonify({"error": "Folder not found"}), 404

    # Get files sorted by modification time (most recent first)
    files = sorted(
        folder_path.glob("*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    file_names = [f.name for f in files]
    return jsonify({"files": file_names})


@app.route("/api/file/<filename>", methods=["GET"])
def get_file(filename):
    """Load a specific file from source folder."""
    source_folder = request.args.get("folder", "v1/labeled")
    file_path = DATA_DIR / source_folder / filename

    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404

    content = file_path.read_text(encoding="utf-8")
    return jsonify({"filename": filename, "content": content})


@app.route("/api/file/<filename>", methods=["POST"])
def save_file(filename):
    """Save/copy a file to target folder and also copy the original document."""
    data = request.json
    source_folder = data.get("source_folder", "v1/labeled")
    target_folder = data.get("target_folder", "v2/labeled")
    content = data.get("content", "")

    # Save the labeled file
    target_path = DATA_DIR / target_folder
    target_path.mkdir(parents=True, exist_ok=True)
    labeled_file_path = target_path / filename
    labeled_file_path.write_text(content, encoding="utf-8")

    # Also copy the corresponding document file
    source_version = source_folder.split("/")[0]  # e.g., "v1"
    target_version = target_folder.split("/")[0]  # e.g., "v2"

    source_doc_path = DATA_DIR / source_version / "documents" / filename
    target_doc_path = DATA_DIR / target_version / "documents" / filename

    if source_doc_path.exists():
        target_doc_path.parent.mkdir(parents=True, exist_ok=True)
        target_doc_path.write_text(source_doc_path.read_text(encoding="utf-8"), encoding="utf-8")

    return jsonify({
        "success": True,
        "filename": filename,
        "saved_to": target_folder,
        "document_copied": source_doc_path.exists()
    })


@app.route("/api/check-exists/<filename>", methods=["GET"])
def check_exists(filename):
    """Check if file exists in target folder."""
    target_folder = request.args.get("folder", "v2/labeled")
    file_path = DATA_DIR / target_folder / filename
    return jsonify({"exists": file_path.exists()})


@app.route("/api/copied-files", methods=["GET"])
def get_copied_files():
    """Get list of all files that exist in target folder."""
    target_folder = request.args.get("folder", "v2/labeled")
    folder_path = DATA_DIR / target_folder

    if not folder_path.exists():
        return jsonify({"files": []})

    # Return just the filenames that exist in the target folder
    existing_files = [f.name for f in folder_path.glob("*.txt")]
    return jsonify({"files": existing_files})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
