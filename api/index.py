from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/status")
def get_status():
    return jsonify({
        "status": "Online",
        "stabilization": "Waiting for data",
        "message": "The Brain is connected!"
    })

if __name__ == "__main__":
    app.run()
