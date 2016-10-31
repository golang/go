// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.6,!go1.8

package gcimporter

import (
	"go/token"
	"go/types"
)

func newAlias(pos token.Pos, pkg *types.Package, name string, orig types.Object) types.Object {
	errorf("unexpected alias in non-Go1.8 export data: %s.%s => %v", pkg.Name(), name, orig)
	panic("unreachable")
}
