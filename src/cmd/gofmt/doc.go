// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Gofmt formats Go programs.

Without an explicit path, it processes the standard input.  Given a file,
it operates on that file; given a directory, it operates on all .go files in
that directory, recursively.  (Files starting with a period are ignored.)

Usage:
	gofmt [flags] [path ...]

The flags are:

	-l
		just list files whose formatting differs from gofmt's; generate no other output
		unless -w is also set.
	-w
		if set, overwrite each input file with its output.
	-spaces
		align with spaces instead of tabs.
	-tabwidth=8
		tab width in spaces.

Debugging flags:

	-trace
		print parse trace.
	-comments=true
		print comments; if false, all comments are elided from the output.

*/
package documentation
