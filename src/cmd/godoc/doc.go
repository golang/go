// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Godoc extracts and generates documentation for Go programs.

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
	-path=""
		additional package directories (colon-separated)
	-html
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

The -path flag accepts a list of colon-separated paths; unrooted paths are relative
to the current working directory. Each path is considered as an additional root for
packages in order of appearance. The last (absolute) path element is the prefix for
the package path. For instance, given the flag value:

	path=".:/home/bar:/public"

for a godoc started in /home/user/godoc, absolute paths are mapped to package paths
as follows:

	/home/user/godoc/x -> godoc/x
	/home/bar/x        -> bar/x
	/public/x          -> public/x

When godoc runs as a web server, it creates a search index from all .go files
under -goroot (excluding files starting with .). The index is created at startup
and is automatically updated every time the -sync command terminates with exit
status 0, indicating that files have changed.

If the sync exit status is 1, godoc assumes that it succeeded without errors
but that no files changed; the index is not updated in this case.

In all other cases, sync is assumed to have failed and godoc backs off running
sync exponentially (up to 1 day). As soon as sync succeeds again (exit status 0
or 1), the normal sync rhythm is re-established.

*/
package documentation
