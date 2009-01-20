// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import Globals "globals"
import GlobalObject "object"
import Type "type"
import Universe "universe"


// ----------------------------------------------------------------------------
// Expressions

const /* op */ (
	LITERAL = iota;
	OBJECT;
	DEREF;
	SELECT;
	CALL;
	TUPLE;
)


// ----------------------------------------------------------------------------
// Literals

type Literal struct {
	pos_ int;
	typ_ *Globals.Type;
	b bool;
	i int;
	f float;
	s string;
}


func (x *Literal) op() int  { return LITERAL; }
func (x *Literal) pos() int  { return x.pos_; }
func (x *Literal) typ() *Globals.Type  { return x.typ_; }


func NewLiteral(pos int, typ *Globals.Type) *Literal {
	x := new(Literal);
	x.pos_ = pos;
	x.typ_ = typ;
	return x;
}


var Bad, True, False, Nil *Literal;


// ----------------------------------------------------------------------------
// Objects

// NOTE We could use Globals.Object directly if we'd added a typ()
// method to its interface. However, this would require renaming the
// typ field everywhere... - Need to think about accessors again.
type Object struct {
	pos_ int;
	obj *Globals.Object;
}


func (x *Object) op() int  { return OBJECT; }
func (x *Object) pos() int  { return x.pos_; }
func (x *Object) typ() *Globals.Type  { return x.obj.typ; }


func NewObject(pos int, obj* Globals.Object) *Object {
	x := new(Object);
	x.pos_ = pos;
	x.obj = obj;
	return x;
}


// ----------------------------------------------------------------------------
// Derefs

// TODO model Deref as unary operation?
type Deref struct {
	ptr_ Globals.Expr;
}


func (x *Deref) op() int  { return DEREF; }
func (x *Deref) pos() int { return x.ptr_.pos(); }
func (x *Deref) typ() *Globals.Type  { return x.ptr_.typ().elt; }


func NewDeref(ptr Globals.Expr) *Deref {
	x := new(Deref);
	x.ptr_ = ptr;
	return x;
}


// ----------------------------------------------------------------------------
// Selectors

// TODO model Selector as binary operation?
type Selector struct {
	pos_ int;
	typ_ *Globals.Type;
}


func (x *Selector) op() int  { return SELECT; }
func (x *Selector) pos() int  { return x.pos_; }
func (x *Selector) typ() *Globals.Type  { return x.typ_; }


func NewSelector(pos int, typ *Globals.Type) *Selector {
	x := new(Selector);
	x.pos_ = pos;
	x.typ_ = typ;
	return x;
}


// ----------------------------------------------------------------------------
// Calls

type Call struct {
	recv, callee Globals.Expr;
	args *Globals.List;
}


func (x *Call) op() int  { return CALL; }
func (x *Call) pos() int  { return 0; }
func (x *Call) typ() *Globals.Type  { return nil; }


func NewCall(args *Globals.List) *Call {
	x := new(Call);
	x.args = args;
	return x;
}


// ----------------------------------------------------------------------------
// Binary expressions

type BinaryExpr struct {
	op_ int;
	pos_ int;
	typ_ *Globals.Type;
	x, y Globals.Expr;
}


func (x *BinaryExpr) op() int  { return x.op_; }
func (x *BinaryExpr) pos() int  { return x.pos_; }
func (x *BinaryExpr) typ() *Globals.Type  {	return x.typ_; }


// ----------------------------------------------------------------------------
// Tuples

type Tuple struct {
	typ_ *Globals.Type;
	list *Globals.List;
}


func (x *Tuple) op() int  {	return TUPLE; }
func (x *Tuple) pos() int  { return x.list.first.expr.pos(); }
func (x *Tuple) typ() *Globals.Type  { return x.typ_; }


func NewTuple(list *Globals.List) *Tuple {
	// make corresponding tuple type
	scope := Globals.NewScope(nil);
	for p := list.first; p != nil; p = p.next {
		x := p.expr;
		obj := Globals.NewObject(x.pos(), GlobalObject.FIELD, "");
		obj.typ = x.typ();
		scope.Add(obj);
	}
	typ := Globals.NewType(Type.TUPLE);
	typ.scope = scope;

	// create the tuple
	x := new(Tuple);
	x.typ_ = typ;
	x.list = list;
	return x;
}


// ----------------------------------------------------------------------------
// Statements

type Block struct {
	// TODO fill in
}


type IfStat struct {
	cond Globals.Expr;
	then_ Globals.Stat;
	else_ Globals.Stat;
}


// ----------------------------------------------------------------------------
// Initialization

func init() {
	Bad = NewLiteral(-1, Universe.bad_t);
	True = NewLiteral(-1, Universe.bool_t);  True.b = true;
	False = NewLiteral(-1, Universe.bool_t);  False.b = false;
	Nil = NewLiteral(-1, Universe.nil_t);
}
