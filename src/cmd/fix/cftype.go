// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(cftypeFix)
}

var cftypeFix = fix{
	name:     "cftype",
	date:     "2017-09-27",
	f:        noop,
	desc:     `Fixes initializers and casts of C.*Ref and JNI types (removed)`,
	disabled: false,
}

func noop(f *ast.File) bool {
	return false
}
