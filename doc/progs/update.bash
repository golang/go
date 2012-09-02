#!/usr/bin/env bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

rm -f *.out *.rej *.orig [568].out

for i in *.go; do
	if grep -q '^// cmpout$' $i; then
		echo $i
		go run $i &> ${i/.go/.out}
	fi
done
