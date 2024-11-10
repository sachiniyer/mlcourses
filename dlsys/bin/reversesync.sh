#!/bin/bash
rsync -avzu --delete -e "ssh" ./dlsys jupyter:~/
