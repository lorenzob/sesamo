#!/bin/bash

nohup ./enrollment-server.sh &
nohup ./training-server.sh &
nohup ./recognition-server.sh &
