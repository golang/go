// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Present displays slide presentations and articles. It runs a web server that
presents slide and article files from the current directory.

It may be run as a stand-alone command or an App Engine app.
The stand-alone version permits the execution of programs from within a
presentation. The App Engine version does not provide this functionality.

Usage of present:
  -base="": base path for slide template and static resources
  -http="127.0.0.1:3999": host:port to listen on

You may use the app.yaml file provided in the root of the go.talks repository
to deploy present to App Engine:
	appcfg.py update -A your-app-id -V your-app-version /path/to/go.talks

Input files are named foo.extension, where "extension" defines the format of
the generated output. The supported formats are:
	.slide        // HTML5 slide presentation
	.article      // article format, such as a blog post

The present file format is documented by the present package:
http://godoc.org/code.google.com/p/go.tools/present
*/
package main
