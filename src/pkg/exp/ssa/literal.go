package ssa

// This file defines the Literal SSA value type.

import (
	"fmt"
	"go/types"
	"math/big"
	"strconv"
)

// newLiteral returns a new literal of the specified value and type.
// val must be valid according to the specification of Literal.Value.
//
func newLiteral(val interface{}, typ types.Type) *Literal {
	// This constructor exists to provide a single place to
	// insert logging/assertions during debugging.
	return &Literal{typ, val}
}

// intLiteral returns an untyped integer literal that evaluates to i.
func intLiteral(i int64) *Literal {
	return newLiteral(i, types.Typ[types.UntypedInt])
}

// nilLiteral returns a nil literal of the specified (reference) type.
func nilLiteral(typ types.Type) *Literal {
	return newLiteral(types.NilType{}, typ)
}

func (l *Literal) Name() string {
	var s string
	switch x := l.Value.(type) {
	case bool:
		s = fmt.Sprintf("%v", l.Value)
	case int64:
		s = fmt.Sprintf("%d", l.Value)
	case *big.Int:
		s = x.String()
	case *big.Rat:
		s = x.FloatString(20)
	case string:
		if len(x) > 20 {
			x = x[:17] + "..." // abbreviate
		}
		s = strconv.Quote(x)
	case types.Complex:
		r := x.Re.FloatString(20)
		i := x.Im.FloatString(20)
		s = fmt.Sprintf("%s+%si", r, i)
	case types.NilType:
		s = "nil"
	default:
		panic(fmt.Sprintf("unexpected literal value: %T", x))
	}
	return s + ":" + l.Type_.String()
}

func (l *Literal) Type() types.Type {
	return l.Type_
}

// IsNil returns true if this literal represents a typed or untyped nil value.
func (l *Literal) IsNil() bool {
	_, ok := l.Value.(types.NilType)
	return ok
}

// Int64 returns the numeric value of this literal truncated to fit
// a signed 64-bit integer.
//
func (l *Literal) Int64() int64 {
	switch x := l.Value.(type) {
	case int64:
		return x
	case *big.Int:
		return x.Int64()
	case *big.Rat:
		// TODO(adonovan): fix: is this the right rounding mode?
		var q big.Int
		return q.Quo(x.Num(), x.Denom()).Int64()
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}

// Uint64 returns the numeric value of this literal truncated to fit
// an unsigned 64-bit integer.
//
func (l *Literal) Uint64() uint64 {
	switch x := l.Value.(type) {
	case int64:
		if x < 0 {
			return 0
		}
		return uint64(x)
	case *big.Int:
		return x.Uint64()
	case *big.Rat:
		// TODO(adonovan): fix: is this right?
		var q big.Int
		return q.Quo(x.Num(), x.Denom()).Uint64()
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}

// Float64 returns the numeric value of this literal truncated to fit
// a float64.
//
func (l *Literal) Float64() float64 {
	switch x := l.Value.(type) {
	case int64:
		return float64(x)
	case *big.Int:
		var r big.Rat
		f, _ := r.SetInt(x).Float64()
		return f
	case *big.Rat:
		f, _ := x.Float64()
		return f
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}

// Complex128 returns the complex value of this literal truncated to
// fit a complex128.
//
func (l *Literal) Complex128() complex128 {
	switch x := l.Value.(type) {
	case int64, *big.Int, *big.Rat:
		return complex(l.Float64(), 0)
	case types.Complex:
		re64, _ := x.Re.Float64()
		im64, _ := x.Im.Float64()
		return complex(re64, im64)
	}
	panic(fmt.Sprintf("unexpected literal value: %T", l.Value))
}
