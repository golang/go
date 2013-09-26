// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.2

package types

import "go/ast"

func slice3(x *ast.SliceExpr) bool {
	return x.Slice3
}

func sliceMax(x *ast.SliceExpr) ast.Expr {
	return x.Max
}
