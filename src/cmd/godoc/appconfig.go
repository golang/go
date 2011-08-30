// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains configuration information used by
// godoc when running on app engine. Adjust as needed
// (typically when the .zip file changes).

package main

const (
	// zipFilename is the name of the .zip file
	// containing the file system served by godoc.
	zipFilename = "godoc.zip"

	// zipGoroot is the path of the goroot directory
	// in the .zip file.
	zipGoroot = "/home/user/go"

	// indexFilenames is a glob pattern specifying
	// files containing the search index served by
	// godoc. The files are concatenated in sorted
	// order (by filename).
	// app-engine limit: file sizes must be <= 10MB;
	// use "split -b8m indexfile index.split." to get
	// smaller files.
	indexFilenames = "index.split.*"
)
