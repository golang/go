package ssa

// This file defines the Literal SSA value type.

import (
	"fmt"
	"go/token"
	"strconv"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// NewLiteral returns a new literal of the specified value and type.
// val must be valid according to the specification of Literal.Value.
//
func NewLiteral(val exact.Value, typ types.Type) *Literal {
	return &Literal{typ, val}
}

// intLiteral returns an untyped integer literal that evaluates to i.
func intLiteral(i int64) *Literal {
	return NewLiteral(exact.MakeInt64(i), types.Typ[types.UntypedInt])
}

// nilLiteral returns a nil literal of the specified type, which may
// be any reference type, including interfaces.
//
func nilLiteral(typ types.Type) *Literal {
	return NewLiteral(exact.MakeNil(), typ)
}

// zeroLiteral returns a new "zero" literal of the specified type,
// which must not be an array or struct type: the zero values of
// aggregates are well-defined but cannot be represented by Literal.
//
func zeroLiteral(t types.Type) *Literal {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return NewLiteral(exact.MakeBool(false), t)
		case t.Info()&types.IsNumeric != 0:
			return NewLiteral(exact.MakeInt64(0), t)
		case t.Info()&types.IsString != 0:
			return NewLiteral(exact.MakeString(""), t)
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return nilLiteral(t)
		default:
			panic(fmt.Sprint("zeroLiteral for unexpected type:", t))
		}
	case *types.Pointer, *types.Slice, *types.Interface, *types.Chan, *types.Map, *types.Signature:
		return nilLiteral(t)
	case *types.Named:
		return NewLiteral(zeroLiteral(t.Underlying()).Value, t)
	case *types.Array, *types.Struct:
		panic(fmt.Sprint("zeroLiteral applied to aggregate:", t))
	}
	panic(fmt.Sprint("zeroLiteral: unexpected ", t))
}

func (l *Literal) Name() string {
	var s string
	if l.Value.Kind() == exact.String {
		s = exact.StringVal(l.Value)
		const maxLit = 20
		if len(s) > maxLit {
			s = s[:maxLit-3] + "..." // abbreviate
		}
		s = strconv.Quote(s)
	} else {
		s = l.Value.String()
	}
	return s + ":" + l.typ.String()
}

func (l *Literal) Type() types.Type {
	return l.typ
}

func (l *Literal) Referrers() *[]Instruction {
	return nil
}

func (l *Literal) Pos() token.Pos {
	// TODO(adonovan): Literals don't have positions.  Should they?
	return token.NoPos
}

// IsNil returns true if this literal represents a typed or untyped nil value.
func (l *Literal) IsNil() bool {
	return l.Value.Kind() == exact.Nil
}

// Int64 returns the numeric value of this literal truncated to fit
// a signed 64-bit integer.
//
func (l *Literal) Int64() int64 {
	switch x := l.Value; x.Kind() {
	case exact.Int:
		if i, ok := exact.Int64Val(x); ok {
			return i
		}
		return 0
	case exact.Float:
		f, _ := exact.Float64Val(x)
		return int64(f)
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}

// Uint64 returns the numeric value of this literal truncated to fit
// an unsigned 64-bit integer.
//
func (l *Literal) Uint64() uint64 {
	switch x := l.Value; x.Kind() {
	case exact.Int:
		if u, ok := exact.Uint64Val(x); ok {
			return u
		}
		return 0
	case exact.Float:
		f, _ := exact.Float64Val(x)
		return uint64(f)
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}

// Float64 returns the numeric value of this literal truncated to fit
// a float64.
//
func (l *Literal) Float64() float64 {
	f, _ := exact.Float64Val(l.Value)
	return f
}

// Complex128 returns the complex value of this literal truncated to
// fit a complex128.
//
func (l *Literal) Complex128() complex128 {
	re, _ := exact.Float64Val(exact.Real(l.Value))
	im, _ := exact.Float64Val(exact.Imag(l.Value))
	return complex(re, im)
}
