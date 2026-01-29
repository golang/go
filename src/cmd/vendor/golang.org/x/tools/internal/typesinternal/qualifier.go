// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/ast"
	"go/types"
	"strconv"
)

// FileQualifier returns a [types.Qualifier] function that qualifies
// imported symbols appropriately based on the import environment of a given
// file.
// If the same package is imported multiple times, the last appearance is
// recorded.
//
// TODO(adonovan): this function ignores the effect of shadowing. It
// should accept a [token.Pos] and a [types.Info] and compute only the
// set of imports that are not shadowed at that point, analogous to
// [analysis.AddImport]. It could also compute (as a side
// effect) the set of additional imports required to ensure that there
// is an accessible import for each necessary package, making it
// converge even more closely with AddImport.
func FileQualifier(f *ast.File, pkg *types.Package) types.Qualifier {
	// Construct mapping of import paths to their defined names.
	// It is only necessary to look at renaming imports.
	imports := make(map[string]string)
	for _, imp := range f.Imports {
		if imp.Name != nil && imp.Name.Name != "_" {
			path, _ := strconv.Unquote(imp.Path.Value)
			imports[path] = imp.Name.Name
		}
	}

	// Define qualifier to replace full package paths with names of the imports.
	return func(p *types.Package) string {
		if p == nil || p == pkg {
			return ""
		}

		if name, ok := imports[p.Path()]; ok {
			if name == "." {
				return ""
			} else {
				return name
			}
		}

		// If there is no local renaming, fall back to the package name.
		return p.Name()
	}
}
