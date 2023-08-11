// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package source

import "go/types"

// containsOrigin reports whether the provided object set contains an object
// with the same origin as the provided obj (which may be a synthetic object
// created during instantiation).
func containsOrigin(objSet map[types.Object]bool, obj types.Object) bool {
	if obj == nil {
		return objSet[obj]
	}
	// In Go 1.18, we can't use the types.Var.Origin and types.Func.Origin methods.
	for target := range objSet {
		if target.Pkg() == obj.Pkg() && target.Pos() == obj.Pos() && target.Name() == obj.Name() {
			return true
		}
	}
	return false
}
