#!/bin/bash
rsync -avzu --delete -e "ssh" jupyter:~/dlsys ./
