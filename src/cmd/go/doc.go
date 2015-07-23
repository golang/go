// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdDoc = &Command{
	Run:         runDoc,
	UsageLine:   "doc [-u] [-c] [package|[package.]symbol[.method]]",
	CustomFlags: true,
	Short:       "show documentation for package or symbol",
	Long: `
Doc prints the documentation comments associated with the item identified by its
arguments (a package, const, func, type, var, or method) followed by a one-line
summary of each of the first-level items "under" that item (package-level
declarations for a package, methods for a type, etc.).

Doc accepts zero, one, or two arguments.

Given no arguments, that is, when run as

	go doc

it prints the package documentation for the package in the current directory.
If the package is a command (package main), the exported symbols of the package
are elided from the presentation unless the -cmd flag is provided.

When run with one argument, the argument is treated as a Go-syntax-like
representation of the item to be documented. What the argument selects depends
on what is installed in GOROOT and GOPATH, as well as the form of the argument,
which is schematically one of these:

	go doc <pkg>
	go doc <sym>[.<method>]
	go doc [<pkg>].<sym>[.<method>]

The first item in this list matched by the argument is the one whose
documentation is printed. (See the examples below.) For packages, the order of
scanning is determined lexically, but the GOROOT tree is always scanned before
GOPATH.

If there is no package specified or matched, the package in the current
directory is selected, so "go doc Foo" shows the documentation for symbol Foo in
the current package.

The package path must be either a qualified path or a proper suffix of a
path. The go tool's usual package mechanism does not apply: package path
elements like . and ... are not implemented by go doc.

When run with two arguments, the first must be a full package path (not just a
suffix), and the second is a symbol or symbol and method; this is similar to the
syntax accepted by godoc:

	go doc <pkg> <sym>[.<method>]

In all forms, when matching symbols, lower-case letters in the argument match
either case but upper-case letters match exactly. This means that there may be
multiple matches of a lower-case argument in a package if different symbols have
different cases. If this occurs, documentation for all matches is printed.

Examples:
	go doc
		Show documentation for current package.
	go doc Foo
		Show documentation for Foo in the current package.
		(Foo starts with a capital letter so it cannot match
		a package path.)
	go doc encoding/json
		Show documentation for the encoding/json package.
	go doc json
		Shorthand for encoding/json.
	go doc json.Number (or go doc json.number)
		Show documentation and method summary for json.Number.
	go doc json.Number.Int64 (or go doc json.number.int64)
		Show documentation for json.Number's Int64 method.
	go doc cmd/doc
		Show package docs for the doc command.
	go doc -cmd cmd/doc
		Show package docs and exported symbols within the doc command.
	go doc template.new
		Show documentation for html/template's New function.
		(html/template is lexically before text/template)
	go doc text/template.new # One argument
		Show documentation for text/template's New function.
	go doc text/template new # Two arguments
		Show documentation for text/template's New function.

Flags:
	-c
		Respect case when matching symbols.
	-cmd
		Treat a command (package main) like a regular package.
		Otherwise package main's exported symbols are hidden
		when showing the package's top-level documentation.
	-u
		Show documentation for unexported as well as exported
		symbols and methods.
`,
}

func runDoc(cmd *Command, args []string) {
	run(buildToolExec, tool("doc"), args)
}
