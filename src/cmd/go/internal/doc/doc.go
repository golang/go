// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc implements the “go doc” command.
package doc

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
)

var CmdDoc = &base.Command{
	Run:         runDoc,
	UsageLine:   "go doc [doc flags] [package|[package.]symbol[.methodOrField]]",
	CustomFlags: true,
	Short:       "show documentation for package or symbol",
	Long: `
Doc prints the documentation comments associated with the item identified by its
arguments (a package, const, func, type, var, method, or struct field)
followed by a one-line summary of each of the first-level items "under"
that item (package-level declarations for a package, methods for a type,
etc.).

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
	go doc <sym>[.<methodOrField>]
	go doc [<pkg>.]<sym>[.<methodOrField>]
	go doc [<pkg>.][<sym>.]<methodOrField>

The first item in this list matched by the argument is the one whose documentation
is printed. (See the examples below.) However, if the argument starts with a capital
letter it is assumed to identify a symbol or method in the current directory.

For packages, the order of scanning is determined lexically in breadth-first order.
That is, the package presented is the one that matches the search and is nearest
the root and lexically first at its level of the hierarchy. The GOROOT tree is
always scanned in its entirety before GOPATH.

If there is no package specified or matched, the package in the current
directory is selected, so "go doc Foo" shows the documentation for symbol Foo in
the current package.

The package path must be either a qualified path or a proper suffix of a
path. The go tool's usual package mechanism does not apply: package path
elements like . and ... are not implemented by go doc.

When run with two arguments, the first is a package path (full path or suffix),
and the second is a symbol, or symbol with method or struct field:

	go doc <pkg> <sym>[.<methodOrField>]

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

	At least in the current tree, these invocations all print the
	documentation for json.Decoder's Decode method:

	go doc json.Decoder.Decode
	go doc json.decoder.decode
	go doc json.decode
	cd go/src/encoding/json; go doc decode

Flags:
	-all
		Show all the documentation for the package.
	-c
		Respect case when matching symbols.
	-cmd
		Treat a command (package main) like a regular package.
		Otherwise package main's exported symbols are hidden
		when showing the package's top-level documentation.
	-short
		One-line representation for each symbol.
	-src
		Show the full source code for the symbol. This will
		display the full Go source of its declaration and
		definition, such as a function definition (including
		the body), type declaration or enclosing const
		block. The output may therefore include unexported
		details.
	-u
		Show documentation for unexported as well as exported
		symbols, methods, and fields.
`,
}

func runDoc(ctx context.Context, cmd *base.Command, args []string) {
	base.StartSigHandlers()
	err := base.RunErr(cfg.BuildToolexec, filepath.Join(cfg.GOROOTbin, "go"), "tool", "doc", args)
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			os.Exit(ee.ExitCode())
		}
		base.Error(err)
	}
}
