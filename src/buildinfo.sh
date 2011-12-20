# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This file is included by shell scripts that need to know the
# full list of architectures, operating systems, and combinations
# that Go runs on.

GOARCHES="
	386
	amd64
	arm
"

GOOSES="
	darwin
	freebsd
	linux
	netbsd
	openbsd
	plan9
	windows
"

GOOSARCHES="
	darwin_386
	darwin_amd64
	freebsd_386
	freebsd_amd64
	linux_386
	linux_amd64
	linux_arm
	netbsd_386
	netbsd_amd64
	openbsd_386
	openbsd_amd64
	plan9_386
	windows_386
	windows_amd64
"
