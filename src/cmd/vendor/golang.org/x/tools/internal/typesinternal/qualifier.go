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
