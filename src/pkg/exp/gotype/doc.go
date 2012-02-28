// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The gotype command does syntactic and semantic analysis of Go files
and packages similar to the analysis performed by the front-end of
a Go compiler. Errors are reported if the analysis fails; otherwise
gotype is quiet (unless -v is set).

Without a list of paths, gotype processes the standard input, which must
be the source of a single package file.

Given a list of file names, each file must be a source file belonging to
the same package unless the package name is explicitly specified with the
-p flag.

Given a directory name, gotype collects all .go files in the directory
and processes them as if they were provided as an explicit list of file
names. Each directory is processed independently. Files starting with .
or not ending in .go are ignored.

Usage:
	gotype [flags] [path ...]

The flags are:
	-e
		Print all (including spurious) errors.
	-p pkgName
		Process only those files in package pkgName.
	-r
		Recursively process subdirectories.
	-v
		Verbose mode.

Debugging flags:
	-comments
		Parse comments (ignored if -ast not set).
	-ast
		Print AST (disables concurrent parsing).
	-trace
		Print parse trace (disables concurrent parsing).


Examples

To check the files file.go, old.saved, and .ignored:

	gotype file.go old.saved .ignored

To check all .go files belonging to package main in the current directory
and recursively in all subdirectories:

	gotype -p main -r .

To verify the output of a pipe:

	echo "package foo" | gotype

*/
package documentation

// BUG(gri): At the moment, only single-file scope analysis is performed.
