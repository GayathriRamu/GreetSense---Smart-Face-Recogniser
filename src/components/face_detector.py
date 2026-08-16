import cv2
import tempfile
import os
from deepface import DeepFace


class FaceDetector:
    """Wraps DeepFace for face detection — tries RetinaFace first,
    falls back to OpenCV if it finds nothing."""
    
    def __init__(self):
        pass

    def detect_faces(self, frame):
        # Write frame to a temp file (closed immediately so DeepFace can read it)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()  # MUST close before cv2/DeepFace can read on macOS/Windows

        try:
            cv2.imwrite(tmp_path, frame)

            # RetinaFace first (best accuracy)
            try:
                results = DeepFace.extract_faces(
                    img_path=tmp_path,
                    detector_backend="retinaface",
                    enforce_detection=False
                )
            except Exception:
                results = []

            # Fallback to opencv if RetinaFace found nothing
            if not results:
                try:
                    results = DeepFace.extract_faces(
                        img_path=tmp_path,           
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                except Exception:
                    results = []

            faces = []
            for res in results:
                facial_area = {}

                if isinstance(res, dict):
                    facial_area = res.get("facial_area", {})

                elif isinstance(res, tuple) and len(res) > 1:
                    region = res[1]
                    if isinstance(region, dict):
                        facial_area = region

                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)

                if w > 0 and h > 0:
                    faces.append((x, y, w, h))

            return faces

        except Exception as e:
            print("Detection error:", e)
            return []

        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame