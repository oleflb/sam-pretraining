#! /usr/bin/env sh

rsync -raP --no-inc-recursive --info=progress2 --exclude-from=.gitignore --exclude=.venv . $1
# rsync -raP --no-inc-recursive --info=progress2 images $1