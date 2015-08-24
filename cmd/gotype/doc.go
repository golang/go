// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The gotype command does syntactic and semantic analysis of Go files
and packages like the front-end of a Go compiler. Errors are reported
if the analysis fails; otherwise gotype is quiet (unless -v is set).

Without a list of paths, gotype reads from standard input, which
must provide a single Go source file defining a complete package.

If a single path is specified that is a directory, gotype checks
the Go files in that directory; they must all belong to the same
package.

Otherwise, each path must be the filename of Go file belonging to
the same package.

Usage:
	gotype [flags] [path...]

The flags are:
	-a
		use all (incl. _test.go) files when processing a directory
	-e
		report all errors (not just the first 10)
	-v
		verbose mode
	-gccgo
		use gccimporter instead of gcimporter

Debugging flags:
	-seq
		parse sequentially, rather than in parallel
	-ast
		print AST (forces -seq)
	-trace
		print parse trace (forces -seq)
	-comments
		parse comments (ignored unless -ast or -trace is provided)

Examples:

To check the files a.go, b.go, and c.go:

	gotype a.go b.go c.go

To check an entire package in the directory dir and print the processed files:

	gotype -v dir

To check an entire package including tests in the local directory:

	gotype -a .

To verify the output of a pipe:

	echo "package foo" | gotype

*/
package main // import "golang.org/x/tools/cmd/gotype"
