// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Buildid displays or updates the build ID stored in a Go package or binary.

Usage:
	go tool buildid [-w] file

By default, buildid prints the build ID found in the named file.
If the -w option is given, buildid rewrites the build ID found in
the file to accurately record a content hash of the file.

This tool is only intended for use by the go command or
other build systems.
*/
package main
