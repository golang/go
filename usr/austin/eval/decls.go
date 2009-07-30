// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
)

/*
 * Types
 */

type Value interface

type Type interface {
	// literal returns this type with all names recursively
	// stripped.  This should only be used when determining
	// assignment compatibility.  To strip a named type for use in
	// a type switch, use .rep().
	literal() Type;
	// rep returns the representative type.  If this is a named
	// type, this is the unnamed underlying type.  Otherwise, this
	// is an identity operation.
	rep() Type;
	// isBoolean returns true if this is a boolean type.
	isBoolean() bool;
	// isInteger returns true if this is an integer type.
	isInteger() bool;
	// isFloat returns true if this is a floating type.
	isFloat() bool;
	// isIdeal returns true if this is an ideal int or float.
	isIdeal() bool;
	// ZeroVal returns a new zero value of this type.
	Zero() Value;
	// String returns the string representation of this type.
	String() string;
}

type BoundedType interface {
	Type;
	// minVal returns the smallest value of this type.
	minVal() *bignum.Rational;
	// maxVal returns the largest value of this type.
	maxVal() *bignum.Rational;
}

/*
 * Values
 */

type Value interface {
	String() string;
	// Assign copies another value into this one.  It should
	// assume that the other value satisfies the same specific
	// value interface (BoolValue, etc.), but must not assume
	// anything about its specific type.
	Assign(o Value);
}

type BoolValue interface {
	Value;
	Get() bool;
	Set(bool);
}

type UintValue interface {
	Value;
	Get() uint64;
	Set(uint64);
}

type IntValue interface {
	Value;
	Get() int64;
	Set(int64);
}

type IdealIntValue interface {
	Value;
	Get() *bignum.Integer;
}

type FloatValue interface {
	Value;
	Get() float64;
	Set(float64);
}

type IdealFloatValue interface {
	Value;
	Get() *bignum.Rational;
}

type StringValue interface {
	Value;
	Get() string;
	Set(string);
}

type ArrayValue interface {
	Value;
	// TODO(austin) Get() is here for uniformity, but is
	// completely useless.  If a lot of other types have similarly
	// useless Get methods, just special-case these uses.
	Get() ArrayValue;
	Elem(i int64) Value;
}

type PtrValue interface {
	Value;
	Get() Value;
	Set(Value);
}

type Func interface
type FuncValue interface {
	Value;
	Get() Func;
	Set(Func);
}

/*
 * Scopes
 */

type Variable struct {
	// Index of this variable in the Frame structure
	Index int;
	// Static type of this variable
	Type Type;
}

type Constant struct {
	Type Type;
	Value Value;
}

// A definition can be a *Variable, *Constant, or Type.
type Def interface {}

type Scope struct

// A block represents a definition block in which a name may not be
// defined more than once.
type block struct {
	// The block enclosing this one, including blocks in other
	// scopes.
	outer *block;
	// The nested block currently being compiled, or nil.
	inner *block;
	// The Scope containing this block.
	scope *Scope;
	// The Variables, Constants, and Types defined in this block.
	defs map[string] Def;
	// The index of the first variable defined in this block.
	// This must be greater than the index of any variable defined
	// in any parent of this block within the same Scope at the
	// time this block is entered.
	offset int;
	// The number of Variables defined in this block.
	numVars int;
}

// A Scope is the compile-time analogue of a Frame, which captures
// some subtree of blocks.
type Scope struct {
	// The root block of this scope.
	*block;
	// The maximum number of variables required at any point in
	// this Scope.  This determines the number of slots needed in
	// Frame's created from this Scope at run-time.
	maxVars int;
}

func (b *block) enterChild() *block
func (b *block) exit()
func (b *block) ChildScope() *Scope
func (b *block) DefineVar(name string, t Type) *Variable
func (b *block) DefineSlot(t Type) *Variable
func (b *block) DefineConst(name string, t Type, v Value) *Constant
func (b *block) DefineType(name string, t Type) Type
func (b *block) Lookup(name string) (level int, def Def)

// The universal scope
func newUniverse() *Scope {
	sc := &Scope{nil, 0};
	sc.block = &block{
		scope: sc,
		defs: make(map[string] Def)
	};
	return sc;
}
var universe *Scope = newUniverse();

/*
 * Frames
 */

type Frame struct {
	Outer *Frame;
	Vars []Value;
}

func (f *Frame) Get(level int, index int) Value
func (f *Frame) child(numVars int) *Frame

func (s *Scope) NewFrame(outer *Frame) *Frame

/*
 * Functions
 */

type Func interface {
	NewFrame() *Frame;
	Call(*Frame);
}
