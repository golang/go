// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// pkgNameOf returns the PkgName associated with the given ImportDecl.
func pkgNameOf(info *types2.Info, decl *syntax.ImportDecl) *types2.PkgName {
	if name := decl.LocalPkgName; name != nil {
		return info.Defs[name].(*types2.PkgName)
	}
	return info.Implicits[decl].(*types2.PkgName)
}
