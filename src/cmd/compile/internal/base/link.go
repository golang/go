// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"cmd/internal/obj"
)

// ReservedImports are import paths used internally for generated
// symbols by the compiler.
//
// The linker uses the magic symbol prefixes "go:" and "type:".
// Avoid potential confusion between import paths and symbols
// by rejecting these reserved imports for now. Also, people
// "can do weird things in GOPATH and we'd prefer they didn't
// do _that_ weird thing" (per rsc). See also #4257.
var ReservedImports = map[string]bool{
	"go":   true,
	"type": true,
}

var Ctxt *obj.Link

// TODO(mdempsky): These should probably be obj.Link methods.

// PkgLinksym returns the linker symbol for name within the given
// package prefix. For user packages, prefix should be the package
// path encoded with objabi.PathToPrefix.
func PkgLinksym(prefix, name string, abi obj.ABI) *obj.LSym {
	if name == "_" {
		// TODO(mdempsky): Cleanup callers and Fatalf instead.
		return linksym(prefix, "_", abi)
	}
	sep := "."
	if ReservedImports[prefix] {
		sep = ":"
	}
	return linksym(prefix, prefix+sep+name, abi)
}

// Linkname returns the linker symbol for the given name as it might
// appear within a //go:linkname directive.
func Linkname(name string, abi obj.ABI) *obj.LSym {
	return linksym("_", name, abi)
}

// linksym is an internal helper function for implementing the above
// exported APIs.
func linksym(pkg, name string, abi obj.ABI) *obj.LSym {
	return Ctxt.LookupABIInit(name, abi, func { r -> r.Pkg = pkg })
}
