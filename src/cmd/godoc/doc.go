// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

The godoc program extracts and generates documentation for Go programs.

It has two modes.

Without the -http flag, it prints plain text documentation to standard output and exits.

	godoc fmt
	godoc fmt Printf

With the -http flag, it runs as a web server and presents the documentation as a web page.

	godoc -http=:6060

Usage:
	godoc [flag] package [name ...]

The flags are:
	-v
		verbose mode
	-tabwidth=4
		width of tabs in units of spaces
	-tmplroot="lib/godoc"
		root template directory (if unrooted, relative to --goroot)
	-pkgroot="src/pkg"
		root package source directory (if unrooted, relative to --goroot)
	-html=
		print HTML in command-line mode
	-goroot=$GOROOT
		Go root directory
	-http=
		HTTP service address (e.g., '127.0.0.1:6060' or just ':6060')
	-sync="command"
		if this and -sync_minutes are set, run the argument as a
		command every sync_minutes; it is intended to update the
		repository holding the source files.
	-sync_minutes=0
		sync interval in minutes; sync is disabled if <= 0

*/
package documentation
