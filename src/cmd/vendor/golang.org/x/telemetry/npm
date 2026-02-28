#!/bin/bash
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

docker run \
  --rm \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  --env NODE_OPTIONS="--dns-result-order=ipv4first" \
  --entrypoint npm \
  node:18.16.0-slim \
  $@
