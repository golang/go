// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import Globals "globals"
import Universe "universe"


// ----------------------------------------------------------------------------
// Expressions

export Expr
type Expr interface {
}


export BinaryExpr
type BinaryExpr struct {
	typ *Globals.Type;
	op int;
	x, y Expr;
}


// ----------------------------------------------------------------------------
// Statements

export Stat
type Stat interface {
}


export Block
type Block struct {
	// TODO fill in
}


export IfStat
type IfStat struct {
	cond Expr;
	then_ Stat;
	else_ Stat;
}
