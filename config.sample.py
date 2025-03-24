# Server config
# Copy this file to config.py and edit accordingly
class ServerConfig:
    FLASK_LISTENING_IP = "127.0.0.1"
    FLASK_PORT = 8000
    FLASK_WORKERS = 4
    FLASK_ROOT = ""
    FLASK_URL_SCHEME = "http"

    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_INDEX = 0

    MFORCE_DIR = "MForce-LTE"
    MFORCE_DATA_SUBDIR = "DATA"

    ADMIN_MAIL_ADDRESS = "dwaipayan.debnath@kuleuven.be"

    BASE_TMP_DIR = "./tmp"
    UPLOAD_SUBDIR = "uploads"