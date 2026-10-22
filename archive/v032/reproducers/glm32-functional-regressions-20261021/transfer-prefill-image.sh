#!/usr/bin/env bash
set -euo pipefail
docker save sha256:84e46c9b33862857d3a47f8c4777b92c73d592fcbc25159c2205b2ee89f56e81 | ssh -o BatchMode=yes node-1 docker load
