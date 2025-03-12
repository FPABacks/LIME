#!/bin/bash

# Make sure the local version of MForce is used.
export MFORCE_DIR="MForce-LTE"
# For the Pleiads the local installations of the software is to be used. 
export PATH=$PATH:/home/frankb/.local/bin:/home/frankb/redis/redis-stable/src/

echo "Path check:"
echo $PATH

# Start redis and celery servers for queue system
echo "Starting Redis server!"
redis-server 6379.conf

echo "Starting Celery"
celery -A app.celery worker --loglevel=info --concurrency=4 --without-gossip --without-mingle --without-heartbeat --max-tasks-per-child=50 &

echo "Starting Gunicorn!"
# Start the WSGI server to get the website up.
gunicorn -w 4 -b 0.0.0.0:8000 app:app
