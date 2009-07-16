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

type Type interface {
	// literal returns this type with all names recursively
	// stripped.
	// TODO(austin) Eliminate the need for this
	literal() Type;
	// compatible returns true if this type is compatible with o.
	// XXX Assignment versus comparison compatibility?
	compatible(o Type) bool;
	// isInteger returns true if this is an integer type.
	isInteger() bool;
	// isFloat returns true if this is a floating type.
	isFloat() bool;
	// isIdeal returns true if this is an ideal int or float.
	isIdeal() bool;
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
	// TODO(austin) Is Type even necessary?
	Type() Type;
	String() string;
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

type PtrValue interface {
	Value;
	Get() Value;
	Set(Value);
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
	// TODO(austin) Need Type?
	Type Type;
	Value Value;
}

// A definition can be a *Variable, *Constant, or Type.
type Def interface {}

type Scope struct {
	outer *Scope;
	defs map[string] Def;
	numVars int;
}

func NewRootScope() *Scope
func (s *Scope) Fork() *Scope
func (s *Scope) DefineVar(name string, t Type) *Variable
func (s *Scope) DefineConst(name string, v Value) *Constant
func (s *Scope) DefineType(name string, t Type) bool
func (s *Scope) Lookup(name string) (Def, *Scope)

/*
 * Frames
 */

type Frame struct {
	Outer *Frame;
	Scope *Scope;
	Vars []Value;
}

func (f *Frame) Get(s *Scope, index int) Value
