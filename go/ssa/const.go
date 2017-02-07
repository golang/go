// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines the Const SSA value type.

import (
	"fmt"
	exact "go/constant"
	"go/token"
	"go/types"
	"strconv"
)

// NewConst returns a new constant of the specified value and type.
// val must be valid according to the specification of Const.Value.
//
func NewConst(val exact.Value, typ types.Type) *Const {
	return &Const{typ, val}
}

// intConst returns an 'int' constant that evaluates to i.
// (i is an int64 in case the host is narrower than the target.)
func intConst(i int64) *Const {
	return NewConst(exact.MakeInt64(i), tInt)
}

// nilConst returns a nil constant of the specified type, which may
// be any reference type, including interfaces.
//
func nilConst(typ types.Type) *Const {
	return NewConst(nil, typ)
}

// stringConst returns a 'string' constant that evaluates to s.
func stringConst(s string) *Const {
	return NewConst(exact.MakeString(s), tString)
}

// zeroConst returns a new "zero" constant of the specified type,
// which must not be an array or struct type: the zero values of
// aggregates are well-defined but cannot be represented by Const.
//
func zeroConst(t types.Type) *Const {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return NewConst(exact.MakeBool(false), t)
		case t.Info()&types.IsNumeric != 0:
			return NewConst(exact.MakeInt64(0), t)
		case t.Info()&types.IsString != 0:
			return NewConst(exact.MakeString(""), t)
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return nilConst(t)
		default:
			panic(fmt.Sprint("zeroConst for unexpected type:", t))
		}
	case *types.Pointer, *types.Slice, *types.Interface, *types.Chan, *types.Map, *types.Signature:
		return nilConst(t)
	case *types.Named:
		return NewConst(zeroConst(t.Underlying()).Value, t)
	case *types.Array, *types.Struct, *types.Tuple:
		panic(fmt.Sprint("zeroConst applied to aggregate:", t))
	}
	panic(fmt.Sprint("zeroConst: unexpected ", t))
}

func (c *Const) RelString(from *types.Package) string {
	var s string
	if c.Value == nil {
		s = "nil"
	} else if c.Value.Kind() == exact.String {
		s = exact.StringVal(c.Value)
		const max = 20
		// TODO(adonovan): don't cut a rune in half.
		if len(s) > max {
			s = s[:max-3] + "..." // abbreviate
		}
		s = strconv.Quote(s)
	} else {
		s = c.Value.String()
	}
	return s + ":" + relType(c.Type(), from)
}

func (c *Const) Name() string {
	return c.RelString(nil)
}

func (c *Const) String() string {
	return c.Name()
}

func (c *Const) Type() types.Type {
	return c.typ
}

func (c *Const) Referrers() *[]Instruction {
	return nil
}

func (c *Const) Parent() *Function { return nil }

func (c *Const) Pos() token.Pos {
	return token.NoPos
}

// IsNil returns true if this constant represents a typed or untyped nil value.
func (c *Const) IsNil() bool {
	return c.Value == nil
}

// TODO(adonovan): move everything below into golang.org/x/tools/go/ssa/interp.

// Int64 returns the numeric value of this constant truncated to fit
// a signed 64-bit integer.
//
func (c *Const) Int64() int64 {
	switch x := exact.ToInt(c.Value); x.Kind() {
	case exact.Int:
		if i, ok := exact.Int64Val(x); ok {
			return i
		}
		return 0
	case exact.Float:
		f, _ := exact.Float64Val(x)
		return int64(f)
	}
	panic(fmt.Sprintf("unexpected constant value: %T", c.Value))
}

// Uint64 returns the numeric value of this constant truncated to fit
// an unsigned 64-bit integer.
//
func (c *Const) Uint64() uint64 {
	switch x := exact.ToInt(c.Value); x.Kind() {
	case exact.Int:
		if u, ok := exact.Uint64Val(x); ok {
			return u
		}
		return 0
	case exact.Float:
		f, _ := exact.Float64Val(x)
		return uint64(f)
	}
	panic(fmt.Sprintf("unexpected constant value: %T", c.Value))
}

// Float64 returns the numeric value of this constant truncated to fit
// a float64.
//
func (c *Const) Float64() float64 {
	f, _ := exact.Float64Val(c.Value)
	return f
}

// Complex128 returns the complex value of this constant truncated to
// fit a complex128.
//
func (c *Const) Complex128() complex128 {
	re, _ := exact.Float64Val(exact.Real(c.Value))
	im, _ := exact.Float64Val(exact.Imag(c.Value))
	return complex(re, im)
}
