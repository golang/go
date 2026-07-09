// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"fmt"
	"iter"
	"reflect"
	"strings"
	"sync"
)

type Expr interface {
	String() string
	eval(b *Bindings) (any, error)
	preorder(yield func(Expr) bool) bool
}

// exprVars yields all Variable nodes in an Expr.
func exprVars(e Expr) iter.Seq[Variable] {
	return func(yield func(Variable) bool) {
		e.preorder(func(e Expr) bool {
			if v, ok := e.(Variable); ok {
				return yield(v)
			}
			return true
		})
	}
}

// Literal is an [Expr] that evaluates to a literal value.
//
// For [Int] and [SymbolicWidth], you probably just want to use those types
// directly. They're literal values, so you could wrap them in a Literal, but
// they are valid expressions on their own.
type Literal struct {
	Val any
}

func (e *Literal) String() string {
	return fmt.Sprint(e.Val)
}
func (e *Literal) eval(b *Bindings) (any, error) {
	if i, ok := e.Val.(int); ok {
		// The evaluator works with Nums, not ints directly.
		return Int(i), nil
	}
	return e.Val, nil
}
func (e *Literal) preorder(yield func(Expr) bool) bool {
	return yield(e)
}

// Variable is an [Expr] that evaluates to the value of the named variable.
type Variable string

func (e Variable) String() string {
	return string(e)
}
func (e Variable) eval(b *Bindings) (any, error) {
	val := b.Get(e)
	if val == nil {
		panic(fmt.Errorf("variable %s not solved", e))
	}
	return val, nil
}
func (e Variable) preorder(yield func(Expr) bool) bool {
	return yield(e)
}

// Func is a function that can be used in an expression. Use the [Func.Apply]
// method to create an [Expr].
type Func struct {
	Name string
	Func func([]any) (any, error)
}

func MakeFunc1[T any](name string, fn func(T) (any, error)) func(e Expr) *Apply {
	f := &Func{
		Name: name,
		Func: func(a []any) (any, error) {
			if len(a) != 1 {
				panic(fmt.Sprintf("%s: got %d arguments, want %d", name, len(a), 1))
			}
			v, ok := a[0].(T)
			if !ok {
				panic(fmt.Sprintf("%s: argument is %T, want %T", name, a[0], *new(T)))
			}
			return fn(v)
		},
	}
	return func(e Expr) *Apply {
		return f.Apply(e)
	}
}
func MakeFunc2[T, U any](name string, fn func(T, U) (any, error)) func(e1, e2 Expr) *Apply {
	f := &Func{
		Name: name,
		Func: func(a []any) (any, error) {
			if len(a) != 2 {
				panic(fmt.Sprintf("%s: got %d arguments, want %d", name, len(a), 2))
			}
			v1, ok := a[0].(T)
			if !ok {
				panic(fmt.Sprintf("%s: argument is %T, want %T", name, a[0], *new(T)))
			}
			v2, ok := a[1].(U)
			if !ok {
				panic(fmt.Sprintf("%s: argument is %T, want %T", name, a[1], *new(U)))
			}
			return fn(v1, v2)
		},
	}
	return func(e1, e2 Expr) *Apply {
		return f.Apply(e1, e2)
	}
}

func MakeFunc(name string, fn func([]any) (any, error)) *Func {
	return &Func{name, fn}
}

type fieldGetterKey struct {
	rt        reflect.Type
	fieldName string
}

var fieldGetters sync.Map

// MakeField returns a Func that projects field fieldName from a value of type
// T. The returned functions are memoized, so only one *Func is created per type
// and field and this is efficient to call repeatedly.
func MakeField[T any](fieldName string) *Func {
	rt := reflect.TypeFor[T]()
	key := fieldGetterKey{rt, fieldName}
	get, ok := fieldGetters.Load(key)
	if !ok {
		f, ok := rt.FieldByName(fieldName)
		if !ok {
			panic(fmt.Sprintf("no such field %s in type %s", fieldName, rt))
		}
		getF := MakeFunc(rt.Name()+"."+fieldName, func(a []any) (any, error) {
			if len(a) != 1 {
				panic("expected exactly 1 argument")
			}
			var rv reflect.Value
			if _, ok := a[0].(T); ok {
				rv = reflect.ValueOf(a[0])
			} else if _, ok := a[0].(*T); ok {
				rv = reflect.ValueOf(a[0]).Elem()
			} else {
				panic(fmt.Sprintf("argument is %T, want %s", a[0], rt))
			}
			return rv.FieldByIndex(f.Index).Interface(), nil
		})
		get, _ = fieldGetters.LoadOrStore(key, getF)
	}
	return get.(*Func)
}

func (f *Func) Apply(args ...Expr) *Apply {
	return &Apply{f, args}
}

// Apply is an [Expr] that applies a [Func] to a sequence of arguments.
type Apply struct {
	Func *Func
	Args []Expr
}

func (e *Apply) String() string {
	var buf strings.Builder
	buf.WriteString(e.Func.Name)
	buf.WriteByte('(')
	for i, x := range e.Args {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(x.String())
	}
	buf.WriteByte(')')
	return buf.String()
}
func (e *Apply) eval(b *Bindings) (any, error) {
	vals := make([]any, 0, 16)
	for _, arg := range e.Args {
		val, err := arg.eval(b)
		if err != nil {
			return nil, err
		}
		vals = append(vals, val)
	}
	return e.Func.Func(vals)
}
func (e *Apply) preorder(yield func(Expr) bool) bool {
	if !yield(e) {
		return false
	}
	for _, a := range e.Args {
		if !a.preorder(yield) {
			return false
		}
	}
	return true
}

// BinExpr is a binary [Expr].
type BinExpr struct {
	Op   BinOp
	X, Y Expr
}

func (e *BinExpr) String() string {
	op := "???"
	if int(e.Op) < len(opStrings) && opStrings[e.Op] != "" {
		op = opStrings[e.Op]
	}
	return e.X.String() + op + e.Y.String()
}
func (e *BinExpr) preorder(yield func(Expr) bool) bool {
	return yield(e) && e.X.preorder(yield) && e.Y.preorder(yield)
}

type BinOp byte

const (
	_       BinOp = iota
	OpTimes       // int = int * int or Width = int * Width (or Width * int)
	OpDiv         // int = int / int (must be exact) or Width = Width / int

	// Comparison operators
	OpEqual          // bool = expr = expr
	OpNotEqual       // bool = expr != expr
	OpGreaterThan    // bool = expr > expr
	OpLessThan       // bool = expr < expr
	OpGreaterOrEqual // bool = expr >= expr
	OpLessOrEqual    // bool = expr <= expr
)

var opStrings = [...]string{
	OpTimes: "*",
	OpDiv:   "/",

	OpEqual:          "=",
	OpNotEqual:       "!=",
	OpGreaterThan:    ">",
	OpLessThan:       "<",
	OpGreaterOrEqual: ">=",
	OpLessOrEqual:    "<=",
}

func (e *BinExpr) eval(b *Bindings) (any, error) {
	xVal, err := e.X.eval(b)
	if err != nil {
		return nil, err
	}
	yVal, err := e.Y.eval(b)
	if err != nil {
		return nil, err
	}

	xn, okX := xVal.(Num)
	yn, okY := yVal.(Num)

	switch e.Op {
	case OpEqual, OpNotEqual:
		if okX && okY {
			// Fall through to numeric operations
			break
		}
		// Otherwise, general equality
		if reflect.TypeOf(xVal) != reflect.TypeOf(yVal) {
			panic(fmt.Errorf("incompatible types for comparison: %T and %T", xVal, yVal))
		}
		if e.Op == OpEqual {
			return xVal == yVal, nil
		} else {
			return xVal != yVal, nil
		}
	}

	if !okX {
		panic(fmt.Errorf("invalid type for %v: %v (%T)", e.Op, xVal, xVal))
	}
	if !okY {
		panic(fmt.Errorf("invalid type for %v: %v (%T)", e.Op, yVal, yVal))
	}

	switch e.Op {
	case OpTimes:
		return xn.Mul(yn)

	case OpDiv:
		return xn.Div(yn)

	case OpEqual, OpNotEqual, OpGreaterThan, OpLessThan, OpGreaterOrEqual, OpLessOrEqual:
		return e.evalComparison(xn, yn)
	}

	panic("bad binop")
}

func (e *BinExpr) evalComparison(x, y Num) (any, error) {
	res, ok := x.Compare(y)
	if !ok {
		// Incomparable
		return e.Op == OpNotEqual, nil
	}
	switch e.Op {
	case OpEqual:
		return res == 0, nil
	case OpNotEqual:
		return res != 0, nil
	case OpGreaterThan:
		return res > 0, nil
	case OpLessThan:
		return res < 0, nil
	case OpGreaterOrEqual:
		return res >= 0, nil
	case OpLessOrEqual:
		return res <= 0, nil
	}
	panic("bad comparison operator")
}
