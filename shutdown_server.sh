#!/usr/bin/env bash

pkill -9 -f celery
redis-cli shutdown