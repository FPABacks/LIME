#!/bin/bash

# Make sure the local version of MForce is used.
export MFORCE_DIR="MForce-LTE"

# Start redis and celery servers for queue system
redis-server &
celery -A app.celery worker --loglevel=info --concurrency=4 --without-gossip --without-mingle --without-heartbeat --max-tasks-per-child=50&

# Start the WSGI server to get the website up.
gunicorn -w 4 -b 0.0.0.0:8000 app:app
