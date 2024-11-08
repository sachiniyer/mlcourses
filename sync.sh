#!/bin/bash

# # Function to clean up background processes on exit
# cleanup() {
#     echo "Stopping rsync and SSH tunnel..."
#     kill "$RSYNC_PID" "$SSH_TUNNEL_PID"
#     exit
# }

# # Trap SIGINT to call cleanup on Ctrl-C
# trap cleanup SIGINT

# # Start reverse SSH tunnel in the background
# ssh -N -L 8888:localhost:8888 jupyter &
# SSH_TUNNEL_PID=$!

# Run rsync in a loop over SSH to keep it active until Ctrl-C
rsync -avzu --delete -e "ssh" jupyter:~/mlcourses/dlsys ./dlsys
