// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Godoc extracts and generates documentation for Go programs.

It has two modes.

Without the -http flag, it runs in command-line mode and prints plain text
documentation to standard output and exits. If both a library package and
a command with the same name exists, using the prefix cmd/ will force
documentation on the command rather than the library package. If the -src
flag is specified, godoc prints the exported interface of a package in Go
source form, or the implementation of a specific exported language entity:

	godoc fmt                # documentation for package fmt
	godoc fmt Printf         # documentation for fmt.Printf
	godoc cmd/go             # force documentation for the go command
	godoc -src fmt           # fmt package interface in Go source form
	godoc -src fmt Printf    # implementation of fmt.Printf

In command-line mode, the -q flag enables search queries against a godoc running
as a webserver. If no explicit server address is specified with the -server flag,
godoc first tries localhost:6060 and then http://golang.org.

	godoc -q Reader
	godoc -q math.Sin
	godoc -server=:6060 -q sin

With the -http flag, it runs as a web server and presents the documentation as a
web page.

	godoc -http=:6060

Usage:

	godoc [flag] package [name ...]

The flags are:

	-v
		verbose mode
	-q
		arguments are considered search queries: a legal query is a
		single identifier (such as ToLower) or a qualified identifier
		(such as math.Sin)
	-src
		print (exported) source in command-line mode
	-tabwidth=4
		width of tabs in units of spaces
	-timestamps=true
		show timestamps with directory listings
	-index
		enable identifier and full text search index
		(no search box is shown if -index is not set)
	-index_files=""
		glob pattern specifying index files; if not empty,
		the index is read from these files in sorted order
	-index_throttle=0.75
		index throttle value; a value of 0 means no time is allocated
		to the indexer (the indexer will never finish), a value of 1.0
		means that index creation is running at full throttle (other
		goroutines may get no time while the index is built)
	-links=true:
		link identifiers to their declarations
	-write_index=false
		write index to a file; the file name must be specified with
		-index_files
	-maxresults=10000
		maximum number of full text search results shown
		(no full text index is built if maxresults <= 0)
	-notes="BUG"
		regular expression matching note markers to show
		(e.g., "BUG|TODO", ".*")
	-html
		print HTML in command-line mode
	-goroot=$GOROOT
		Go root directory
	-http=addr
		HTTP service address (e.g., '127.0.0.1:6060' or just ':6060')
	-server=addr
		webserver address for command line searches
	-analysis=type,pointer
		comma-separated list of analyses to perform
		"type": display identifier resolution, type info, method sets,
			'implements', and static callees
		"pointer": display channel peers, callers and dynamic callees
			(significantly slower)
		See http://golang.org/lib/godoc/analysis/help.html for details.
	-templates=""
		directory containing alternate template files; if set,
		the directory may provide alternative template files
		for the files in $GOROOT/lib/godoc
	-url=path
		print to standard output the data that would be served by
		an HTTP request for path
	-zip=""
		zip file providing the file system to serve; disabled if empty

By default, godoc looks at the packages it finds via $GOROOT and $GOPATH (if set).
This behavior can be altered by providing an alternative $GOROOT with the -goroot
flag.

When godoc runs as a web server and -index is set, a search index is maintained.
The index is created at startup.

The index contains both identifier and full text search information (searchable
via regular expressions). The maximum number of full text search results shown
can be set with the -maxresults flag; if set to 0, no full text results are
shown, and only an identifier index but no full text search index is created.

By default, godoc uses the system's GOOS/GOARCH; in command-line mode you can
set the GOOS/GOARCH environment variables to get output for the system specified.
If -http was specified you can provide the URL parameters "GOOS" and "GOARCH"
to set the output on the web page.

The presentation mode of web pages served by godoc can be controlled with the
"m" URL parameter; it accepts a comma-separated list of flag names as value:

	all	show documentation for all declarations, not just the exported ones
	methods	show all embedded methods, not just those of unexported anonymous fields
	src	show the original source code rather then the extracted documentation
	text	present the page in textual (command-line) form rather than HTML
	flat	present flat (not indented) directory listings using full paths

For instance, http://golang.org/pkg/math/big/?m=all,text shows the documentation
for all (not just the exported) declarations of package big, in textual form (as
it would appear when using godoc from the command line: "godoc -src math/big .*").

By default, godoc serves files from the file system of the underlying OS.
Instead, a .zip file may be provided via the -zip flag, which contains
the file system to serve. The file paths stored in the .zip file must use
slash ('/') as path separator; and they must be unrooted. $GOROOT (or -goroot)
must be set to the .zip file directory path containing the Go root directory.
For instance, for a .zip file created by the command:

	zip -r go.zip $HOME/go

one may run godoc as follows:

	godoc -http=:6060 -zip=go.zip -goroot=$HOME/go

Godoc documentation is converted to HTML or to text using the go/doc package;
see http://golang.org/pkg/go/doc/#ToHTML for the exact rules.
Godoc also shows example code that is runnable by the testing package;
see http://golang.org/pkg/testing/#hdr-Examples for the conventions.
See "Godoc: documenting Go code" for how to write good comments for godoc:
http://golang.org/doc/articles/godoc_documenting_go_code.html

*/
package main // import "golang.org/x/tools/cmd/godoc"
