"""All Video Intelligence API features run on a video stored on GCS."""
import os
import base64
import time
from google.cloud import videointelligence

# Decode service account key from env and save to a temp file
base64_key = os.environ.get("GCS_KEY_BASE64")
key_path = "/tmp/gcs-key.json"

with open(key_path, "wb") as f:
    f.write(base64.b64decode(base64_key))

# Get bucket name and construct URIs
bucket_name = os.environ.get("GCS_BUCKET_NAME")
input_filename = "YOURVIDEO.mp4"  # You can override this dynamically later
gcs_uri = f"gs://{bucket_name}/{input_filename}"
output_uri = f"gs://{bucket_name}/output-{int(time.time())}.json"

# Create client from service account key
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

# Advanced configs
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

# Run video intelligence
operation = video_client.annotate_video(
    request={
        "features": features,
        "input_uri": gcs_uri,
        "output_uri": output_uri,
        "video_context": video_context
    }
)

print("\nProcessing video:", gcs_uri)
result = operation.result(timeout=300)
print("\nFinished processing.")
