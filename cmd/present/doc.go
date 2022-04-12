// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Present displays slide presentations and articles. It runs a web server that
presents slide and article files from the current directory.

It may be run as a stand-alone command or an App Engine app.

The setup of the Go version of NaCl is documented at:
https://golang.org/wiki/NativeClient

To use with App Engine, copy the files in the tools/cmd/present directory to the
root of your application and create an app.yaml file similar to this:

	runtime: go111

	handlers:
	- url: /favicon.ico
	  static_files: static/favicon.ico
	  upload: static/favicon.ico
	- url: /static
	  static_dir: static
	- url: /.*
	  script: auto

	# nobuild_files is a regexp that identifies which files to not build.  It
	# is useful for embedding static assets like code snippets and preventing
	# them from producing build errors for your project.
	nobuild_files: [path regexp for talk materials]

When running on App Engine, content will be served from the ./content/
subdirectory.

Present then can be tested in a local App Engine environment with

	GAE_ENV=standard go run .

And deployed using

	gcloud app deploy

Input files are named foo.extension, where "extension" defines the format of
the generated output. The supported formats are:

	.slide        // HTML5 slide presentation
	.article      // article format, such as a blog post

The present file format is documented by the present package:
https://pkg.go.dev/golang.org/x/tools/present
*/
package main
