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
	-r rule
		apply the rewrite rule to the source before reformatting.
	-w
		if set, overwrite each input file with its output.
	-spaces
		align with spaces instead of tabs.
	-tabindent
		indent with tabs independent of -spaces.
	-tabwidth=8
		tab width in spaces.

Flags to aid the transition to the new semicolon-free syntax (these flags will be
removed eventually):

	-oldparser=true
		parse old syntax (required semicolons).
	-oldprinter=true
		print old syntax (required semicolons).

Debugging flags:

	-trace
		print parse trace.
	-comments=true
		print comments; if false, all comments are elided from the output.

The rewrite rule specified with the -r flag must be a string of the form:

	pattern -> replacement

Both pattern and replacement must be valid Go expressions.
In the pattern, single-character lowercase identifers serve as
wildcards matching arbitrary subexpressions; those expressions
will be substituted for the same identifiers in the replacement.


Examples

To check files for unnecessary parentheses:

	gofmt -r '(a) -> a' -l *.go

To remove the parentheses:

	gofmt -r '(a) -> a' -w *.go

To convert the package tree from explicit slice upper bounds to implicit ones:

	gofmt -r 'α[β:len(α)] -> α[β:]' -w $GOROOT/src/pkg
*/
package documentation

// BUG(rsc): The implementation of -r is a bit slow.
