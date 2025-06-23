// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(contextFix)
}

var contextFix = fix{
	name:     "context",
	date:     "2016-09-09",
	f:        ctxfix,
	desc:     `Change imports of golang.org/x/net/context to context`,
	disabled: false,
}

func ctxfix(f *ast.File) bool {
	return rewriteImport(f, "golang.org/x/net/context", "context")
}
