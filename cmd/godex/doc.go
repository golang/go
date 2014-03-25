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
// Usage: godex [flags] {path|qualifiedIdent}
//
// Each argument must be a package path, or a qualified identifier.
//
// The flags are:
//
//	-s=src
//		only consider packages from src, where src is one of the supported compilers
//	-v
//		verbose mode
//
// The following sources are supported:
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
// TODO(gri) expand this documentation
//
package main
