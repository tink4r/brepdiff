#!/bin/bash
# entrypoint.sh

# Initialize micromamba shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate occenv

# Symlink setup (if needed)
[ -e ./psr ] || ln -s /brepdiff/psr ./psr
[ -e ./blender ] || ln -s /brepdiff/blender ./blender

# Start interactive shell
exec bash
