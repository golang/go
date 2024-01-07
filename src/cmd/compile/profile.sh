# Copyright 2023 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script collects a CPU profile of the compiler
# for building all targets in std and cmd, and puts
# the profile at cmd/compile/default.pgo.

dir=$(mktemp -d)
cd $dir
seed=$(date)

for p in $(go list std cmd); do
	h=$(echo $seed $p | md5sum | cut -d ' ' -f 1)
	echo $p $h
	go build -o /dev/null -gcflags=-cpuprofile=$PWD/prof.$h $p
done

go tool pprof -proto prof.* > $(go env GOROOT)/src/cmd/compile/default.pgo

rm -r $dir
