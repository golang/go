// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

type InterpExpr struct {
	Parts  []string
	Exprs  []Expr
	Opens  []Pos
	Closes []Pos
	Raw    string
	expr
}
