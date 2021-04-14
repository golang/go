// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(eglFixDisplay)
	register(eglFixConfig)
}

var eglFixDisplay = fix{
	name:     "egl",
	date:     "2018-12-15",
	f:        eglfixDisp,
	desc:     `Fixes initializers of EGLDisplay`,
	disabled: false,
}

// Old state:
//   type EGLDisplay unsafe.Pointer
// New state:
//   type EGLDisplay uintptr
// This fix finds nils initializing these types and replaces the nils with 0s.
func eglfixDisp(f *ast.File) bool {
	return typefix(f, func(s string) bool {
		return s == "C.EGLDisplay"
	})
}

var eglFixConfig = fix{
	name:     "eglconf",
	date:     "2020-05-30",
	f:        eglfixConfig,
	desc:     `Fixes initializers of EGLConfig`,
	disabled: false,
}

// Old state:
//   type EGLConfig unsafe.Pointer
// New state:
//   type EGLConfig uintptr
// This fix finds nils initializing these types and replaces the nils with 0s.
func eglfixConfig(f *ast.File) bool {
	return typefix(f, func(s string) bool {
		return s == "C.EGLConfig"
	})
}
