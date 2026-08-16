import cv2
import numpy as np


class FacePreprocessor:

    def extract_face(self, frame, bbox):
        # clamp bbox to frame bounds so we don't crop outside the image
        # (can happen with detections near the edge)
        x, y, w, h = bbox

        height, width, _ = frame.shape

        # Ensure bbox is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        face = frame[y:y2, x:x2]

        # to handle cases where bbox is invalid or results in empty face
        if face is None or face.size == 0:
            return None

        return face

    def preprocess_face(self, face, target_size=(160, 160)):
        # resize + BGR->RGB since that's what the embedding models expect
        try:
            # Resize
            face = cv2.resize(face, target_size)

            # Convert to RGB (DeepFace expects RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Normalize pixel values
            # face = face.astype("float32") / 255.0

            return face

        except Exception as e:
            print("Preprocessing error:", e)
            return None