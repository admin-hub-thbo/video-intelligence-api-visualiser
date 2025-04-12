import os
import base64
import time
from flask import Flask, request, jsonify
from google.cloud import videointelligence

# Setup Flask app
app = Flask(__name__)

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
        # Decode service account key from env and save to a temp file
        base64_key = os.environ.get("GCS_KEY_BASE64")
        key_path = "/tmp/gcs-key.json"
        with open(key_path, "wb") as f:
            f.write(base64.b64decode(base64_key))

        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        gcs_uri = f"gs://{bucket_name}/{input_filename}"
        output_uri = f"gs://{bucket_name}/output-{int(time.time())}.json"

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
                "input_uri": gcs_uri,
                "output_uri": output_uri,
                "video_context": video_context
            }
        )

        print(f"Processing video: {gcs_uri}")
        result = operation.result(timeout=300)
        print("Finished processing.")

        return jsonify({
            "message": "Video annotation complete",
            "input_uri": gcs_uri,
            "output_uri": output_uri
        })

    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": str(e)}), 500

# Render requires this
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
