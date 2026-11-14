#!/usr/bin/env bash
# FVP (Fluid Video Player): download and install (or update) in one go, no root needed:
#   curl -fsSL https://raw.githubusercontent.com/kupecjan1980-pixel/fvp/master/get.sh | bash
set -euo pipefail
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL https://github.com/kupecjan1980-pixel/fvp/archive/refs/heads/master.tar.gz | tar -xz -C "$TMP"
bash "$TMP"/fvp-master/install.sh
