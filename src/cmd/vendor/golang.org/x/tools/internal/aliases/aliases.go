// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aliases

import (
	"go/token"
	"go/types"
)

// New creates a new TypeName in Package pkg that
// is an alias for the type rhs.
func New(pos token.Pos, pkg *types.Package, name string, rhs types.Type, tparams []*types.TypeParam) *types.TypeName {
	tname := types.NewTypeName(pos, pkg, name, nil)
	types.NewAlias(tname, rhs).SetTypeParams(tparams)
	return tname
}
