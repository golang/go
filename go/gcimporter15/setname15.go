// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5,!go1.6

package gcimporter

import (
	"go/types"
	"unsafe"
)

func setName(pkg *types.Package, name string) {
	(*types_Package)(unsafe.Pointer(pkg)).name = name
}

// The underlying type of types_Package is identical to
// the underlying type of types.Package. We use it with
// package unsafe to set the name field since 1.5 does
// not have the Package.SetName method.
// TestSetName verifies that the layout with respect to
// the name field is correct.
type types_Package struct {
	path     string
	name     string
	scope    *types.Scope
	complete bool
	imports  []*types.Package
	fake     bool
}
