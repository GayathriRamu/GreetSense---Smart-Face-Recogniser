from deepface import DeepFace
import numpy as np
import time
import cv2


class FaceRecognizer:
    def __init__(self, models=None):
        # default to Facenet + ArcFace if nothing's passed in
        if models is None:
            self.models = ["Facenet", "ArcFace"]
        else:
            self.models = models

    def get_embeddings(self, face):
        """Runs the face through each configured model and returns
        {model_name: embedding} plus timing per model (used by the
        evaluator to report avg inference time)."""
        embeddings = {}
        timings = {}

        try:
            # Handling tuple input
            if isinstance(face, tuple):
                face = face[0]

            # Validating image
            if face is None or not hasattr(face, "shape"):
                return {}, {}

            # Ensure correct format 
            if len(face.shape) != 3:
                return {}, {}

            # Convert to RGB (DeepFace expects RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print("Pre-check error:", e)
            return {}, {}

        for model_name in self.models:
            try:
                start = time.time()

                result = DeepFace.represent(
                    img_path=face,
                    model_name=model_name,
                    enforce_detection=False
                )

                if isinstance(result, list) and len(result) > 0:
                    embedding = np.array(result[0].get("embedding", []))
                else:
                    embedding = None

                embeddings[model_name] = embedding
                timings[model_name] = time.time() - start

            except Exception as e:
                print(f"{model_name} error:", e)
                embeddings[model_name] = None
                timings[model_name] = None

        return embeddings, timings

    def recognize(self, embedding, users_db, model_name):
        """Compares against every stored embedding for that model and
        returns the closest match by cosine distance, or "Unknown" if
        nothing's under the threshold."""

        if embedding is None:
            return "Unknown", float("inf")

        embedding = embedding / np.linalg.norm(embedding)

        best_distance = float("inf")
        best_match = "Unknown"

        for user_name, db_data in users_db.items():

            if model_name not in db_data:
                continue

            for db_emb in db_data[model_name]:
                db_embedding = np.array(db_emb)
                db_embedding = db_embedding / np.linalg.norm(db_embedding)

                distance = 1 - np.dot(embedding, db_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_match = user_name

        thresholds = {
            "Facenet": 0.40,
            "ArcFace": 0.30
        }

        if best_distance < thresholds.get(model_name, 0.30):
            return best_match, best_distance
        else:
            return "Unknown", best_distance