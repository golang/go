// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(eglFix)
}

var eglFix = fix{
	name:     "egl",
	date:     "2018-12-15",
	f:        eglfix,
	desc:     `Fixes initializers of EGLDisplay`,
	disabled: false,
}

// Old state:
//   type EGLDisplay unsafe.Pointer
// New state:
//   type EGLDisplay uintptr
// This fix finds nils initializing these types and replaces the nils with 0s.
func eglfix(f *ast.File) bool {
	return typefix(f, func(s string) bool {
		return s == "C.EGLDisplay"
	})
}
