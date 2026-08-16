import json
import os


class DatabaseHandler:
    # simple JSON-backed store: {user_name: {model_name: [embeddings]}}
    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(__file__),
            "../database/users.json"
        )

        # Ensure file exists
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def get_all_users(self):
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except:
            return {}

    def save_all_users(self, data):
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    # save function 
    def save_user(self, name, embedding, model_name):
        # appends rather than overwrites, so a user can have multiple
        # embeddings per model (helps recognition accuracy)
        users = self.get_all_users()

        if name not in users:
            users[name] = {}

        if model_name not in users[name]:
            users[name][model_name] = []

        # store multiple embeddings
        users[name][model_name].append(embedding.tolist())

        self.save_all_users(users)