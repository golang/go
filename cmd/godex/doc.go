// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The godex command prints (dumps) exported information of packages
// or selected package objects.
//
// In contrast to godoc, godex extracts this information from compiled
// object files. Hence the exported data is truly what a compiler will
// see, at the cost of missing commentary.
//
// Usage: godex [flags] {path[.name]}
//
// Each argument must be a (possibly partial) package path, optionally
// followed by a dot and the name of a package object:
//
//	godex math
//	godex math.Sin
//	godex math.Sin fmt.Printf
//	godex go/types
//
// godex automatically tries all possible package path prefixes if only a
// partial package path is given. For instance, for the path "go/types",
// godex prepends "golang.org/x/tools".
//
// The prefixes are computed by searching the directories specified by
// the GOROOT and GOPATH environment variables (and by excluding the
// build OS- and architecture-specific directory names from the path).
// The search order is depth-first and alphabetic; for a partial path
// "foo", a package "a/foo" is found before "b/foo".
//
// Absolute and relative paths may be provided, which disable automatic
// prefix generation:
//
//	godex $GOROOT/pkg/darwin_amd64/sort
//	godex ./sort
//
// All but the last path element may contain dots; a dot in the last path
// element separates the package path from the package object name. If the
// last path element contains a dot, terminate the argument with another
// dot (indicating an empty object name). For instance, the path for a
// package foo.bar would be specified as in:
//
//	godex foo.bar.
//
// The flags are:
//
//	-s=""
//		only consider packages from src, where src is one of the supported compilers
//	-v=false
//		verbose mode
//
// The following sources (-s arguments) are supported:
//
//	gc
//		gc-generated object files
//	gccgo
//		gccgo-generated object files
//	gccgo-new
//		gccgo-generated object files using a condensed format (experimental)
//	source
//		(uncompiled) source code (not yet implemented)
//
// If no -s argument is provided, godex will try to find a matching source.
package main // import "golang.org/x/tools/cmd/godex"

// BUG(gri): support for -s=source is not yet implemented
// BUG(gri): gccgo-importing appears to have occasional problems stalling godex; try -s=gc as work-around
