#!/bin/bash

# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

if ! command -v gsutil 2>&1 > /dev/null; then
  echo "Install gsutil:"
  echo
  echo "   https://cloud.google.com/storage/docs/gsutil_install#sdk-install"
fi

if [ ! -d build ]; then
  echo "Run make.bash first"
fi

set -e -o -x

gsutil -m cp -a public-read build/* gs://golang/getgo
