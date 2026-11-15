// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type OptionType struct {
	Elem Expr
	expr
}

type CoalesceExpr struct {
	X, Y Expr
	expr
}

type OptSelectExpr struct {
	X   Expr
	Sel *Name
	expr
}
