import os
import base64
import time
import yt_dlp
from flask import Flask, request, jsonify
from google.cloud import videointelligence
from google.cloud import storage

# Setup Flask app
app = Flask(__name__)

# Cache for the key path
_KEY_PATH = "/tmp/gcs-key.json"

def ensure_key_file():
    if not os.path.exists(_KEY_PATH):
        base64_key = os.environ.get("GCS_KEY_BASE64")
        if base64_key:
            with open(_KEY_PATH, "wb") as f:
                f.write(base64.b64decode(base64_key))
    return _KEY_PATH

def get_gcs_client():
    key_path = ensure_key_file()
    return storage.Client.from_service_account_json(key_path)

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    # Return both the gs:// URI and the public https:// URL
    return f"gs://{bucket_name}/{destination_blob_name}", f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"

def run_annotation(input_uri, output_uri):
    key_path = ensure_key_file()
    # Create client
    video_client = videointelligence.VideoIntelligenceServiceClient.from_service_account_file(key_path)

    # Full feature list
    features = [
        videointelligence.Feature.OBJECT_TRACKING,
        videointelligence.Feature.LABEL_DETECTION,
        videointelligence.Feature.SHOT_CHANGE_DETECTION,
        videointelligence.Feature.SPEECH_TRANSCRIPTION,
        videointelligence.Feature.LOGO_RECOGNITION,
        videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
        videointelligence.Feature.TEXT_DETECTION,
        videointelligence.Feature.FACE_DETECTION,
        videointelligence.Feature.PERSON_DETECTION
    ]

    transcript_config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )
    person_config = videointelligence.PersonDetectionConfig(
        include_bounding_boxes=True,
        include_attributes=False,
        include_pose_landmarks=True,
    )
    face_config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True, include_attributes=True
    )

    video_context = videointelligence.VideoContext(
        speech_transcription_config=transcript_config,
        person_detection_config=person_config,
        face_detection_config=face_config
    )

    # Annotate the video
    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "video_context": video_context
        }
    )

    print(f"Processing video: {input_uri}")
    result = operation.result(timeout=600)
    print("Finished processing.")
    return result

@app.route("/")
def index():
    return "Video Intelligence API Helper is live."

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    input_filename = data.get("filename")
    if not input_filename:
        return jsonify({"error": "Missing filename"}), 400

    try:
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        gcs_uri = f"gs://{bucket_name}/{input_filename}"
        output_uri = f"gs://{bucket_name}/output-{int(time.time())}.json"

        run_annotation(gcs_uri, output_uri)

        return jsonify({
            "message": "Video annotation complete",
            "input_uri": gcs_uri,
            "output_uri": output_uri,
            "json_url": f"https://storage.googleapis.com/{bucket_name}/{os.path.basename(output_uri)}"
        })

    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/process-url", methods=["POST"])
def process_url():
    data = request.json
    video_url = data.get("url")
    if not video_url:
        return jsonify({"error": "Missing URL"}), 400

    try:
        bucket_name = os.environ.get("GCS_BUCKET_NAME")

        timestamp = int(time.time())
        local_filename = f"video-{timestamp}.mp4"
        local_path = os.path.join("/tmp", local_filename)

        # Download video using yt-dlp
        ydl_opts = {
            'outtmpl': local_path,
            'format': 'best[ext=mp4]/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Upload to GCS
        gcs_uri, public_video_url = upload_to_gcs(local_path, bucket_name, local_filename)

        # Output URI
        output_filename = f"output-{timestamp}.json"
        output_uri = f"gs://{bucket_name}/{output_filename}"
        public_json_url = f"https://storage.googleapis.com/{bucket_name}/{output_filename}"

        # Annotate
        run_annotation(gcs_uri, output_uri)

        # Cleanup local file
        if os.path.exists(local_path):
            os.remove(local_path)

        return jsonify({
            "message": "Video URL processing and annotation complete",
            "video_url": video_url,
            "public_video_url": public_video_url,
            "public_json_url": public_json_url,
            "input_uri": gcs_uri,
            "output_uri": output_uri
        })

    except Exception as e:
        print("Error during URL processing:", str(e))
        return jsonify({"error": str(e)}), 500

# Render requires this
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
