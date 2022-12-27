// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines the Const SSA value type.

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"strconv"
	"strings"

	"golang.org/x/tools/internal/typeparams"
)

// NewConst returns a new constant of the specified value and type.
// val must be valid according to the specification of Const.Value.
func NewConst(val constant.Value, typ types.Type) *Const {
	if val == nil {
		switch soleTypeKind(typ) {
		case types.IsBoolean:
			val = constant.MakeBool(false)
		case types.IsInteger:
			val = constant.MakeInt64(0)
		case types.IsString:
			val = constant.MakeString("")
		}
	}
	return &Const{typ, val}
}

// soleTypeKind returns a BasicInfo for which constant.Value can
// represent all zero values for the types in the type set.
//
//	types.IsBoolean for false is a representative.
//	types.IsInteger for 0
//	types.IsString for ""
//	0 otherwise.
func soleTypeKind(typ types.Type) types.BasicInfo {
	// State records the set of possible zero values (false, 0, "").
	// Candidates (perhaps all) are eliminated during the type-set
	// iteration, which executes at least once.
	state := types.IsBoolean | types.IsInteger | types.IsString
	underIs(typeSetOf(typ), func(t types.Type) bool {
		var c types.BasicInfo
		if t, ok := t.(*types.Basic); ok {
			c = t.Info()
		}
		if c&types.IsNumeric != 0 { // int/float/complex
			c = types.IsInteger
		}
		state = state & c
		return state != 0
	})
	return state
}

// intConst returns an 'int' constant that evaluates to i.
// (i is an int64 in case the host is narrower than the target.)
func intConst(i int64) *Const {
	return NewConst(constant.MakeInt64(i), tInt)
}

// stringConst returns a 'string' constant that evaluates to s.
func stringConst(s string) *Const {
	return NewConst(constant.MakeString(s), tString)
}

// zeroConst returns a new "zero" constant of the specified type.
func zeroConst(t types.Type) *Const {
	return NewConst(nil, t)
}

func (c *Const) RelString(from *types.Package) string {
	var s string
	if c.Value == nil {
		s = zeroString(c.typ, from)
	} else if c.Value.Kind() == constant.String {
		s = constant.StringVal(c.Value)
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

// zeroString returns the string representation of the "zero" value of the type t.
func zeroString(t types.Type, from *types.Package) string {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return "false"
		case t.Info()&types.IsNumeric != 0:
			return "0"
		case t.Info()&types.IsString != 0:
			return `""`
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return "nil"
		default:
			panic(fmt.Sprint("zeroString for unexpected type:", t))
		}
	case *types.Pointer, *types.Slice, *types.Interface, *types.Chan, *types.Map, *types.Signature:
		return "nil"
	case *types.Named:
		return zeroString(t.Underlying(), from)
	case *types.Array, *types.Struct:
		return relType(t, from) + "{}"
	case *types.Tuple:
		// Tuples are not normal values.
		// We are currently format as "(t[0], ..., t[n])". Could be something else.
		components := make([]string, t.Len())
		for i := 0; i < t.Len(); i++ {
			components[i] = zeroString(t.At(i).Type(), from)
		}
		return "(" + strings.Join(components, ", ") + ")"
	case *typeparams.TypeParam:
		return "*new(" + relType(t, from) + ")"
	}
	panic(fmt.Sprint("zeroString: unexpected ", t))
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

// IsNil returns true if this constant is a nil value of
// a nillable reference type (pointer, slice, channel, map, or function),
// a basic interface type, or
// a type parameter all of whose possible instantiations are themselves nillable.
func (c *Const) IsNil() bool {
	return c.Value == nil && nillable(c.typ)
}

// nillable reports whether *new(T) == nil is legal for type T.
func nillable(t types.Type) bool {
	if typeparams.IsTypeParam(t) {
		return underIs(typeSetOf(t), func(u types.Type) bool {
			// empty type set (u==nil) => any underlying types => not nillable
			return u != nil && nillable(u)
		})
	}
	switch t.Underlying().(type) {
	case *types.Pointer, *types.Slice, *types.Chan, *types.Map, *types.Signature:
		return true
	case *types.Interface:
		return true // basic interface.
	default:
		return false
	}
}

// TODO(adonovan): move everything below into golang.org/x/tools/go/ssa/interp.

// Int64 returns the numeric value of this constant truncated to fit
// a signed 64-bit integer.
func (c *Const) Int64() int64 {
	switch x := constant.ToInt(c.Value); x.Kind() {
	case constant.Int:
		if i, ok := constant.Int64Val(x); ok {
			return i
		}
		return 0
	case constant.Float:
		f, _ := constant.Float64Val(x)
		return int64(f)
	}
	panic(fmt.Sprintf("unexpected constant value: %T", c.Value))
}

// Uint64 returns the numeric value of this constant truncated to fit
// an unsigned 64-bit integer.
func (c *Const) Uint64() uint64 {
	switch x := constant.ToInt(c.Value); x.Kind() {
	case constant.Int:
		if u, ok := constant.Uint64Val(x); ok {
			return u
		}
		return 0
	case constant.Float:
		f, _ := constant.Float64Val(x)
		return uint64(f)
	}
	panic(fmt.Sprintf("unexpected constant value: %T", c.Value))
}

// Float64 returns the numeric value of this constant truncated to fit
// a float64.
func (c *Const) Float64() float64 {
	x := constant.ToFloat(c.Value) // (c.Value == nil) => x.Kind() == Unknown
	f, _ := constant.Float64Val(x)
	return f
}

// Complex128 returns the complex value of this constant truncated to
// fit a complex128.
func (c *Const) Complex128() complex128 {
	x := constant.ToComplex(c.Value) // (c.Value == nil) => x.Kind() == Unknown
	re, _ := constant.Float64Val(constant.Real(x))
	im, _ := constant.Float64Val(constant.Imag(x))
	return complex(re, im)
}
