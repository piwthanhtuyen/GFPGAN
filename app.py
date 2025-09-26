import os
import sys
import subprocess
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Map version -> output folder (bên trong static/results/)
OUTPUT_DIRS = {
    "ver1": os.path.join("static", "results", "v1"),      # tô màu
    "ver2": os.path.join("static", "results", "vclean"),  # làm sạch
    "ver3": os.path.join("static", "results", "v13"),     # rõ nét
    "ver4": os.path.join("static", "results", "v14"),     # đánh bóng
}


@app.route("/", methods=["GET", "POST"])
def index():
    before_file, after_file, version = None, None, None

    if request.method == "POST":
        if "file" not in request.files:
            return "Không có file upload!"

        file = request.files["file"]
        if file.filename == "":
            return "Chưa chọn file!"

        version = request.form.get("version")
        scale = request.form.get("scale", "2")  # mặc định scale = 2

        if version not in OUTPUT_DIRS:
            return "Sai version!"

        # Thư mục input
        input_dir = "inputs"
        os.makedirs(input_dir, exist_ok=True)
        filepath = os.path.join(input_dir, file.filename)
        file.save(filepath)

        # Thư mục output
        output_dir = OUTPUT_DIRS[version]
        os.makedirs(output_dir, exist_ok=True)

        # Script inference
        script_path = os.path.join(os.path.dirname(__file__), "inference_gfpgan.py")

        # Lệnh chạy GFP-GAN
        cmd = [
            sys.executable, script_path,
            "-i", input_dir,
            "-o", output_dir,
            "-v", "1.2",
            "-s", scale
        ]

        print(" Đang chạy lệnh:", " ".join(cmd))

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            return f"""
            ⚠ Lỗi khi chạy GFP-GAN:<br>
            <b>CMD:</b> {" ".join(e.cmd)}<br>
            <b>Return code:</b> {e.returncode}<br>
            <b>STDOUT:</b><pre>{e.stdout}</pre><br>
            <b>STDERR:</b><pre>{e.stderr}</pre>
            """

        before_file = file.filename
        after_file = file.filename

    return render_template("index.html",
                           uploaded_file=before_file,
                           result_file=after_file,
                           version=version)


@app.route("/inputs/<filename>")
def uploaded_file(filename):
    return send_from_directory("inputs", filename)


@app.route("/results/<version>/<filename>")
def result_file(version, filename):
    folder = OUTPUT_DIRS[version]
    return send_from_directory(folder, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
