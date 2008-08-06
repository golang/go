// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import Globals "globals"
import Universe "universe"


// ----------------------------------------------------------------------------
// Expressions


export type Literal struct {
	typ_ *Globals.Type;
	b bool;
	i int;
	f float;
	s string;
}


func (x *Literal) typ() *Globals.Type {
	return x.typ_;
}


export func NewLiteral(typ *Globals.Type) *Literal {
	x := new(Literal);
	x.typ_ = typ;
	return x;
}


export var Bad, True, False, Nil *Literal;


// NOTE We could use Globals.Object directly if we'd added a typ()
// method to its interface. However, this would require renaming the
// typ field everywhere... - Need to think about accessors again.
export type Object struct {
	obj *Globals.Object;
}


func (x *Object) typ() *Globals.Type {
	return x.obj.typ;
}


export func NewObject(obj* Globals.Object) *Object {
	x := new(Object);
	x.obj = obj;
	return x;
}


export type Selector struct {
	typ_ *Globals.Type;
}


func (x *Selector) typ() *Globals.Type {
	return x.typ_;
}


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


// ----------------------------------------------------------------------------
// Initialization

func init() {
	Bad = NewLiteral(Universe.bad_t);
	True = NewLiteral(Universe.bool_t);  True.b = true;
	False = NewLiteral(Universe.bool_t);  False.b = false;
	Nil = NewLiteral(Universe.nil_t);
}
