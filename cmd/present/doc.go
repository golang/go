// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Present displays slide presentations and articles. It runs a web server that
presents slide and article files from the current directory.

It may be run as a stand-alone command or an App Engine app.
Instructions for deployment to App Engine are in the README of the
golang.org/x/tools repository.

Usage of present:
  -base="": base path for slide template and static resources
  -http="127.0.0.1:3999": HTTP service address (e.g., '127.0.0.1:3999')
  -nacl=false: use Native Client environment playground (prevents non-Go code execution)
  -orighost="": host component of web origin URL (e.g., 'localhost')
  -play=true: enable playground (permit execution of arbitrary user code)

The setup of the Go version of NaCl is documented at:
https://golang.org/wiki/NativeClient

Input files are named foo.extension, where "extension" defines the format of
the generated output. The supported formats are:
	.slide        // HTML5 slide presentation
	.article      // article format, such as a blog post

The present file format is documented by the present package:
http://godoc.org/golang.org/x/tools/present
*/
package main // import "golang.org/x/tools/cmd/present"
