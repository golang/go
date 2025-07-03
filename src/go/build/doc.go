// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package build gathers information about Go packages.
//
// # Build Constraints
//
// A build constraint, also known as a build tag, is a condition under which a
// file should be included in the package. Build constraints are given by a
// line comment that begins
//
//	//go:build
//
// Build constraints may also be part of a file's name
// (for example, source_windows.go will only be included if the target
// operating system is windows).
//
// See 'go help buildconstraint'
// (https://pkg.go.dev/cmd/go#hdr-Build_constraints) for details.
//
// # Go Path
//
// The Go path is a list of directory trees containing Go source code.
// It is consulted to resolve imports that cannot be found in the standard
// Go tree. The default path is the value of the GOPATH environment
// variable, interpreted as a path list appropriate to the operating system
// (on Unix, the variable is a colon-separated string;
// on Windows, a semicolon-separated string;
// on Plan 9, a list).
//
// Each directory listed in the Go path must have a prescribed structure:
//
// The src/ directory holds source code. The path below 'src' determines
// the import path or executable name.
//
// The pkg/ directory holds installed package objects.
// As in the Go tree, each target operating system and
// architecture pair has its own subdirectory of pkg
// (pkg/GOOS_GOARCH).
//
// If DIR is a directory listed in the Go path, a package with
// source in DIR/src/foo/bar can be imported as "foo/bar" and
// has its compiled form installed to "DIR/pkg/GOOS_GOARCH/foo/bar.a"
// (or, for gccgo, "DIR/pkg/gccgo/foo/libbar.a").
//
// The bin/ directory holds compiled commands.
// Each command is named for its source directory, but only
// using the final element, not the entire path. That is, the
// command with source in DIR/src/foo/quux is installed into
// DIR/bin/quux, not DIR/bin/foo/quux. The foo/ is stripped
// so that you can add DIR/bin to your PATH to get at the
// installed commands.
//
// Here's an example directory layout:
//
//	GOPATH=/home/user/gocode
//
//	/home/user/gocode/
//	    src/
//	        foo/
//	            bar/               (go code in package bar)
//	                x.go
//	            quux/              (go code in package main)
//	                y.go
//	    bin/
//	        quux                   (installed command)
//	    pkg/
//	        linux_amd64/
//	            foo/
//	                bar.a          (installed package object)
//
// # Binary-Only Packages
//
// In Go 1.12 and earlier, it was possible to distribute packages in binary
// form without including the source code used for compiling the package.
// The package was distributed with a source file not excluded by build
// constraints and containing a "//go:binary-only-package" comment. Like a
// build constraint, this comment appeared at the top of a file, preceded
// only by blank lines and other line comments and with a blank line
// following the comment, to separate it from the package documentation.
// Unlike build constraints, this comment is only recognized in non-test
// Go source files.
//
// The minimal source code for a binary-only package was therefore:
//
//	//go:binary-only-package
//
//	package mypkg
//
// The source code could include additional Go code. That code was never
// compiled but would be processed by tools like godoc and might be useful
// as end-user documentation.
//
// "go build" and other commands no longer support binary-only-packages.
// [Import] and [ImportDir] will still set the BinaryOnly flag in packages
// containing these comments for use in tools and error messages.
package build
