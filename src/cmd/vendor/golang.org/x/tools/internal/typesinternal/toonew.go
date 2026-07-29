// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/types"

	"golang.org/x/tools/internal/stdlib"
	"golang.org/x/tools/internal/versions"
)

// TooNewStdSymbols computes the set of package-level symbols
// exported by pkg that are not available at the specified version.
//
// The pkg is allowed to contain type errors.
func TooNewStdSymbols(pkg *types.Package, version string) map[types.Object]stdlib.Symbol {
	disallowed := make(map[types.Object]stdlib.Symbol)

	// Some symbols are accessible before their release but
	// only with specific build tags unknown to us here.
	// Avoid false positives in such cases.
	if pkg.Path() == "testing/synctest" && versions.AtLeast(version, "go1.24") {
		// requires go1.24 && goexperiment.synctest || go1.25
		return disallowed
	}

	// Pass 1: package-level symbols.
	symbols := stdlib.PackageSymbols[pkg.Path()]
	for _, sym := range symbols {
		if versions.Before(version, sym.Version.String()) {
			switch sym.Kind {
			case stdlib.Func, stdlib.Var, stdlib.Const, stdlib.Type:
				disallowed[pkg.Scope().Lookup(sym.Name)] = sym
			}
		}
	}

	// Pass 2: fields and methods.
	//
	// We allow fields and methods if their associated type is
	// disallowed, as otherwise we would report false positives
	// for compatibility shims. Consider:
	//
	//   //go:build go1.22
	//   type T struct { F std.Real } // correct new API
	//
	//   //go:build !go1.22
	//   type T struct { F fake } // shim
	//   type fake struct { ... }
	//   func (fake) M () {}
	//
	// These alternative declarations of T use either the std.Real
	// type, introduced in go1.22, or a fake type, for the field
	// F. (The fakery could be arbitrarily deep, involving more
	// nested fields and methods than are shown here.) Clients
	// that use the compatibility shim T will compile with any
	// version of go, whether older or newer than go1.22, but only
	// the newer version will use the std.Real implementation.
	//
	// Now consider a reference to method M in new(T).F.M() in a
	// module that requires a minimum of go1.21. The analysis may
	// occur using a version of Go higher than 1.21, selecting the
	// first version of T, so the method M is Real.M. This would
	// spuriously cause the analyzer to report a reference to a
	// too-new symbol even though this expression compiles just
	// fine (with the fake implementation) using go1.21.
	var noSym stdlib.Symbol
	depth := make(map[types.Object]int)
	for _, sym := range symbols {
		if !versions.Before(version, sym.Version.String()) {
			continue // allowed
		}

		var obj types.Object
		var indices []int
		switch sym.Kind {
		case stdlib.Field:
			typename, name := sym.SplitField()
			if t := pkg.Scope().Lookup(typename); t != nil && disallowed[t] == noSym {
				obj, indices, _ = types.LookupFieldOrMethod(t.Type(), false, pkg, name)
			}

		case stdlib.Method:
			ptr, recvname, name := sym.SplitMethod()
			if t := pkg.Scope().Lookup(recvname); t != nil && disallowed[t] == noSym {
				obj, indices, _ = types.LookupFieldOrMethod(t.Type(), ptr, pkg, name)
			}
		}
		if obj != nil {
			// In the presence of embedding, two or more "pkg.T.name"
			// strings may map to the same types.Object.
			// Prefer the Object with the shorter index path.
			if min, ok := depth[obj]; !ok || len(indices) < min {
				depth[obj] = len(indices)
				disallowed[obj] = sym
			}
		}
	}

	return disallowed
}
