// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cmdDoc = &Command{
	Run:         runDoc,
	UsageLine:   "doc [-u] [package|[package.]symbol[.method]]",
	CustomFlags: true,
	Short:       "show documentation for package or symbol",
	Long: `
Doc accepts at most one argument, indicating either a package, a symbol within a
package, or a method of a symbol.

	go doc
	go doc <pkg>
	go doc <sym>[.<method>]
	go doc [<pkg>].<sym>[.<method>]

Doc interprets the argument to see what it represents, determined by its syntax
and which packages and symbols are present in the source directories of GOROOT and
GOPATH.

The first item in this list that succeeds is the one whose documentation is printed.
For packages, the order of scanning is determined lexically, however the GOROOT
tree is always scanned before GOPATH.

If there is no package specified or matched, the package in the current directory
is selected, so "go doc" shows the documentation for the current package and
"go doc Foo" shows the documentation for symbol Foo in the current package.

Doc prints the documentation comments associated with the top-level item the
argument identifies (package, type, method) followed by a one-line summary of each
of the first-level items "under" that item (package-level declarations for a
package, methods for a type, etc.).

The package paths must be either a qualified path or a proper suffix of a path
(see examples below). The go tool's usual package mechanism does not apply: package
path elements like . and ... are not implemented by go doc.

When matching symbols, lower-case letters match either case but upper-case letters
match exactly.

Examples:
	go doc
		Show documentation for current package.
	go doc Foo
		Show documentation for Foo in the current package.
		(Foo starts with a capital letter so it cannot match a package path.)
	go doc encoding/json
		Show documentation for the encoding/json package.
	go doc json
		Shorthand for encoding/json.
	go doc json.Number (or go doc json.number)
		Show documentation and method summary for json.Number.
	go doc json.Number.Int64 (or go doc json.number.int64)
		Show documentation for json.Number's Int64 method.

Flags:
	-u
		Show documentation for unexported as well as exported
		symbols and methods.
`,
}

func runDoc(cmd *Command, args []string) {
	run(buildToolExec, tool("doc"), args)
}
