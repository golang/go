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
//      godex go/types
//
// All but the last path element may contain dots. godex automatically
// tries all possible package path prefixes for non-standard library
// packages if only a partial package path is given. For instance, for
// the path "go/types", godex prepends "code.google.com/p/go.tools".
//
// The prefixes are computed by searching the directories specified by
// the GOPATH environment variable (and by excluding the build os and
// architecture specific directory names from the path). The search
// order is depth-first and alphabetic; for a partial path "foo", a
// package "a/foo" is found before "b/foo".
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
//
package main

// BUG(gri) std-library packages should also benefit from auto-generated prefixes.
