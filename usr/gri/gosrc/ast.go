// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import Globals "globals"
import Universe "universe"


// ----------------------------------------------------------------------------
// Expressions

export type BinaryExpr struct {
	typ_ *Globals.Type;
	op int;
	x, y Globals.Expr;
}



func (x *BinaryExpr) typ() *Globals.Type {
	return x.typ_;
}


// ----------------------------------------------------------------------------
// Statements

export type Block struct {
	// TODO fill in
}


export type IfStat struct {
	cond Globals.Expr;
	then_ Globals.Stat;
	else_ Globals.Stat;
}
