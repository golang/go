// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"flag"
	"fmt"
	"go/token"
	"log"
	"os"
	"reflect"
	"regexp"
	"testing"
)

// All tests are done using the same file set.
var fset = token.NewFileSet()

// Print each statement or expression before parsing it
var noisy = false

func init() { flag.BoolVar(&noisy, "noisy", false, "chatter during eval tests") }

/*
 * Generic statement/expression test framework
 */

type test []job

type job struct {
	code  string
	cerr  string
	rterr string
	val   Value
	noval bool
}

func runTests(t *testing.T, baseName string, tests []test) {
	for i, test := range tests {
		name := fmt.Sprintf("%s[%d]", baseName, i)
		test.run(t, name)
	}
}

func (a test) run(t *testing.T, name string) {
	w := newTestWorld()
	for _, j := range a {
		src := j.code + ";" // trailing semicolon to finish statement
		if noisy {
			println("code:", src)
		}

		code, err := w.Compile(fset, src)
		if err != nil {
			if j.cerr == "" {
				t.Errorf("%s: Compile %s: %v", name, src, err)
				break
			}
			if !match(t, err, j.cerr) {
				t.Errorf("%s: Compile %s = error %s; want %v", name, src, err, j.cerr)
				break
			}
			continue
		}
		if j.cerr != "" {
			t.Errorf("%s: Compile %s succeeded; want %s", name, src, j.cerr)
			break
		}

		val, err := code.Run()
		if err != nil {
			if j.rterr == "" {
				t.Errorf("%s: Run %s: %v", name, src, err)
				break
			}
			if !match(t, err, j.rterr) {
				t.Errorf("%s: Run %s = error %s; want %v", name, src, err, j.rterr)
				break
			}
			continue
		}
		if j.rterr != "" {
			t.Errorf("%s: Run %s succeeded; want %s", name, src, j.rterr)
			break
		}

		if !j.noval && !reflect.DeepEqual(val, j.val) {
			t.Errorf("%s: Run %s = %T(%v) want %T(%v)", name, src, val, val, j.val, j.val)
		}
	}
}

func match(t *testing.T, err os.Error, pat string) bool {
	ok, err1 := regexp.MatchString(pat, err.String())
	if err1 != nil {
		t.Fatalf("compile regexp %s: %v", pat, err1)
	}
	return ok
}


/*
 * Test constructors
 */

// Expression compile error
func CErr(expr string, cerr string) test { return test([]job{{code: expr, cerr: cerr}}) }

// Expression runtime error
func RErr(expr string, rterr string) test { return test([]job{{code: expr, rterr: rterr}}) }

// Expression value
func Val(expr string, val interface{}) test {
	return test([]job{{code: expr, val: toValue(val)}})
}

// Statement runs without error
func Run(stmts string) test { return test([]job{{code: stmts, noval: true}}) }

// Two statements without error.
// TODO(rsc): Should be possible with Run but the parser
// won't let us do both top-level and non-top-level statements.
func Run2(stmt1, stmt2 string) test {
	return test([]job{{code: stmt1, noval: true}, {code: stmt2, noval: true}})
}

// Statement runs and test one expression's value
func Val1(stmts string, expr1 string, val1 interface{}) test {
	return test([]job{
		{code: stmts, noval: true},
		{code: expr1, val: toValue(val1)},
	})
}

// Statement runs and test two expressions' values
func Val2(stmts string, expr1 string, val1 interface{}, expr2 string, val2 interface{}) test {
	return test([]job{
		{code: stmts, noval: true},
		{code: expr1, val: toValue(val1)},
		{code: expr2, val: toValue(val2)},
	})
}

/*
 * Value constructors
 */

type vstruct []interface{}

type varray []interface{}

type vslice struct {
	arr      varray
	len, cap int
}

func toValue(val interface{}) Value {
	switch val := val.(type) {
	case bool:
		r := boolV(val)
		return &r
	case uint8:
		r := uint8V(val)
		return &r
	case uint:
		r := uintV(val)
		return &r
	case int:
		r := intV(val)
		return &r
	case *big.Int:
		return &idealIntV{val}
	case float64:
		r := float64V(val)
		return &r
	case *big.Rat:
		return &idealFloatV{val}
	case string:
		r := stringV(val)
		return &r
	case vstruct:
		elems := make([]Value, len(val))
		for i, e := range val {
			elems[i] = toValue(e)
		}
		r := structV(elems)
		return &r
	case varray:
		elems := make([]Value, len(val))
		for i, e := range val {
			elems[i] = toValue(e)
		}
		r := arrayV(elems)
		return &r
	case vslice:
		return &sliceV{Slice{toValue(val.arr).(ArrayValue), int64(val.len), int64(val.cap)}}
	case Func:
		return &funcV{val}
	}
	log.Panicf("toValue(%T) not implemented", val)
	panic("unreachable")
}

/*
 * Default test scope
 */

type testFunc struct{}

func (*testFunc) NewFrame() *Frame { return &Frame{nil, make([]Value, 2)} }

func (*testFunc) Call(t *Thread) {
	n := t.f.Vars[0].(IntValue).Get(t)

	res := n + 1

	t.f.Vars[1].(IntValue).Set(t, res)
}

type oneTwoFunc struct{}

func (*oneTwoFunc) NewFrame() *Frame { return &Frame{nil, make([]Value, 2)} }

func (*oneTwoFunc) Call(t *Thread) {
	t.f.Vars[0].(IntValue).Set(t, 1)
	t.f.Vars[1].(IntValue).Set(t, 2)
}

type voidFunc struct{}

func (*voidFunc) NewFrame() *Frame { return &Frame{nil, []Value{}} }

func (*voidFunc) Call(t *Thread) {}

func newTestWorld() *World {
	w := NewWorld()

	def := func(name string, t Type, val interface{}) { w.DefineVar(name, t, toValue(val)) }

	w.DefineConst("c", IdealIntType, toValue(big.NewInt(1)))
	def("i", IntType, 1)
	def("i2", IntType, 2)
	def("u", UintType, uint(1))
	def("f", Float64Type, 1.0)
	def("s", StringType, "abc")
	def("t", NewStructType([]StructField{{"a", IntType, false}}), vstruct{1})
	def("ai", NewArrayType(2, IntType), varray{1, 2})
	def("aai", NewArrayType(2, NewArrayType(2, IntType)), varray{varray{1, 2}, varray{3, 4}})
	def("aai2", NewArrayType(2, NewArrayType(2, IntType)), varray{varray{5, 6}, varray{7, 8}})
	def("fn", NewFuncType([]Type{IntType}, false, []Type{IntType}), &testFunc{})
	def("oneTwo", NewFuncType([]Type{}, false, []Type{IntType, IntType}), &oneTwoFunc{})
	def("void", NewFuncType([]Type{}, false, []Type{}), &voidFunc{})
	def("sli", NewSliceType(IntType), vslice{varray{1, 2, 3}, 2, 3})

	return w
}
