// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.6,!go1.8

package gcimporter

import (
	"go/token"
	"go/types"
)

type types_Alias struct {
	types.Object
	dummy int
} // satisfies types.Object but will never be encountered

func types_NewAlias(pos token.Pos, pkg *types.Package, name string, orig types.Object) types.Object {
	errorf("unexpected alias in non-Go1.8 export data: %s.%s => %v", pkg.Name(), name, orig) // panics
	panic("unreachable")
}

func original(types.Object) types.Object {
	panic("unreachable")
}

const testfile = "exports17.go"
