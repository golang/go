// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Doc (usually run as go doc) accepts zero, one or two arguments.
//
// Zero arguments:
//
//	go doc
//
// Show the documentation for the package in the current directory.
//
// One argument:
//
//	go doc <pkg>
//	go doc <sym>[.<methodOrField>]
//	go doc [<pkg>.]<sym>[.<methodOrField>]
//	go doc [<pkg>.][<sym>.]<methodOrField>
//
// The first item in this list that succeeds is the one whose documentation
// is printed. If there is a symbol but no package, the package in the current
// directory is chosen. However, if the argument begins with a capital
// letter it is always assumed to be a symbol in the current directory.
//
// Two arguments:
//
//	go doc <pkg> <sym>[.<methodOrField>]
//
// Show the documentation for the package, symbol, and method or field. The
// first argument must be a full package path. This is similar to the
// command-line usage for the godoc command.
//
// For commands, unless the -cmd flag is present "go doc command"
// shows only the package-level docs for the package.
//
// The -src flag causes doc to print the full source code for the symbol, such
// as the body of a struct, function or method.
//
// The -all flag causes doc to print all documentation for the package and
// all its visible symbols. The argument must identify a package.
//
// For complete documentation, run "go help doc".
package main

import (
	"cmd/internal/doc"
	"cmd/internal/telemetry/counter"
	"os"
)

func main() {
	counter.Open()
	counter.Inc("doc/invocations")
	doc.Main(os.Args[1:])
}
