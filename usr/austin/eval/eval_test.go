// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"fmt";
	"go/parser";
	"go/scanner";
	"go/token";
	"log";
	"os";
	"reflect";
	"testing";
)

// Print each statement or expression before parsing it
const noisy = false

/*
 * Generic statement/expression test framework
 */

type test struct {
	code string;
	rterr string;
	exprs []exprTest;
	cerr string;
}

type exprTest struct {
	code string;
	val interface{};
	rterr string;
}

func runTests(t *testing.T, baseName string, tests []test) {
	for i, test := range tests {
		name := fmt.Sprintf("%s[%d]", baseName, i);
		test.run(t, name);
	}
}

func (a *test) run(t *testing.T, name string) {
	sc := newTestScope();

	var fr *Frame;
	var cerr os.Error;

	if a.code != "" {
		if noisy {
			println(a.code);
		}

		// Compile statements
		asts, err := parser.ParseStmtList(name, a.code);
		if err != nil && cerr == nil {
			cerr = err;
		}
		code, err := CompileStmts(sc, asts);
		if err != nil && cerr == nil {
			cerr = err;
		}

		// Execute statements
		if cerr == nil {
			fr = sc.NewFrame(nil);
			rterr := code.Exec(fr);
			if a.rterr == "" && rterr != nil {
				t.Errorf("%s: expected %s to run, got runtime error %v", name, a.code, rterr);
				return;
			} else if !checkRTError(t, name, a.code, rterr, a.rterr) {
				return;
			}
		}
	}

	if fr == nil {
		fr = sc.NewFrame(nil);
	}
	for _, e := range a.exprs {
		if cerr != nil {
			break;
		}

		if noisy {
			println(e.code);
		}

		// Compile expression
		ast, err := parser.ParseExpr(name, e.code);
		if err != nil && cerr == nil {
			cerr = err;
		}
		code, err := CompileExpr(sc, ast);
		if err != nil && cerr == nil {
			cerr = err;
		}

		// Evaluate expression
		if cerr == nil {
			val, rterr := code.Eval(fr);
			if e.rterr == "" && rterr != nil {
				t.Errorf("%s: expected %q to have value %T(%v), got runtime error %v", name, e.code, e.val, e.val, rterr);
			} else if !checkRTError(t, name, e.code, rterr, e.rterr) {
				continue;
			}
			if e.val != nil {
				wantval := toValue(e.val);
				if !reflect.DeepEqual(val, wantval) {
					t.Errorf("%s: expected %q to have value %T(%v), got %T(%v)", name, e.code, wantval, wantval, val, val);
				}
			}
		}
	}

	// Check compile errors
	switch {
	case cerr == nil && a.cerr == "":
		// Good
	case cerr == nil && a.cerr != "":
		t.Errorf("%s: expected compile error matching %q, got no errors", name, a.cerr);
	case cerr != nil && a.cerr == "":
		t.Errorf("%s: expected no compile error, got error %v", name, cerr);
	case cerr != nil && a.cerr != "":
		cerr := cerr.(scanner.ErrorList);
		if len(cerr) > 1 {
			t.Errorf("%s: expected 1 compile error matching %q, got %v", name, a.cerr, cerr);
			break;
		}
		m, err := testing.MatchString(a.cerr, cerr.String());
		if err != "" {
			t.Fatalf("%s: failed to compile regexp %q: %s", name, a.cerr, err);
		}
		if !m {
			t.Errorf("%s: expected compile error matching %q, got compile error %v", name, a.cerr, cerr);
		}
	}
}

func checkRTError(t *testing.T, name string, code string, rterr os.Error, pat string) bool {
	switch {
	case rterr == nil && pat == "":
		return true;
		
	case rterr == nil && pat != "":
		t.Errorf("%s: expected %s to fail with runtime error matching %q, got no error", name, code, pat);
		return false;

	case rterr != nil && pat != "":
		m, err := testing.MatchString(pat, rterr.String());
		if err != "" {
			t.Fatalf("%s: failed to compile regexp %q: %s", name, pat, err);
		}
		if !m {
			t.Errorf("%s: expected runtime error matching %q, got runtime error %v", name, pat, rterr);
			return false;
		}
		return true;
	}
	panic("rterr != nil && pat == \"\" should have been handled by the caller");
}

/*
 * Test constructors
 */

// Expression compile error
func EErr(expr string, cerr string) test {
	return test{"", "", []exprTest{exprTest{expr, nil, ""}}, cerr};
}

// Expression runtime error
func ERTErr(expr string, rterr string) test {
	return test{"", "", []exprTest{exprTest{expr, nil, rterr}}, ""};
}

// Expression value
func Val(expr string, val interface{}) test {
	return test{"", "", []exprTest{exprTest{expr, val, ""}}, ""};
}

// Statement compile error
func SErr(stmts string, cerr string) test {
	return test{stmts, "", nil, cerr};
}

// Statement runtime error
func SRTErr(stmts string, rterr string) test {
	return test{stmts, rterr, nil, ""};
}

// Statement runs without error
func SRuns(stmts string) test {
	return test{stmts, "", nil, ""};
}

// Statement runs and test one expression's value
func Val1(stmts string, expr1 string, val1 interface{}) test {
	return test{stmts, "", []exprTest{exprTest{expr1, val1, ""}}, ""};
}

// Statement runs and test two expressions' values
func Val2(stmts string, expr1 string, val1 interface{}, expr2 string, val2 interface{}) test {
	return test{stmts, "", []exprTest{exprTest{expr1, val1, ""}, exprTest{expr2, val2, ""}}, ""};
}

/*
 * Value constructors
 */

type vstruct []interface{}

type varray []interface{}

type vslice struct {
	arr varray;
	len, cap int;
}

func toValue(val interface{}) Value {
	switch val := val.(type) {
	case bool:
		r := boolV(val);
		return &r;
	case uint8:
		r := uint8V(val);
		return &r;
	case uint:
		r := uintV(val);
		return &r;
	case int:
		r := intV(val);
		return &r;
	case *bignum.Integer:
		return &idealIntV{val};
	case float:
		r := floatV(val);
		return &r;
	case *bignum.Rational:
		return &idealFloatV{val};
	case string:
		r := stringV(val);
		return &r;
	case vstruct:
		elems := make([]Value, len(val));
		for i, e := range val {
			elems[i] = toValue(e);
		}
		r := structV(elems);
		return &r;
	case varray:
		elems := make([]Value, len(val));
		for i, e := range val {
			elems[i] = toValue(e);
		}
		r := arrayV(elems);
		return &r;
	case vslice:
		return &sliceV{Slice{toValue(val.arr).(ArrayValue), int64(val.len), int64(val.cap)}};
	case Func:
		return &funcV{val};
	}
	log.Crashf("toValue(%T) not implemented", val);
	panic();
}

/*
 * Default test scope
 */

type testFunc struct {};

func (*testFunc) NewFrame() *Frame {
	return &Frame{nil, &[2]Value {}};
}

func (*testFunc) Call(fr *Frame) {
	n := fr.Vars[0].(IntValue).Get();

	res := n + 1;

	fr.Vars[1].(IntValue).Set(res);
}

type oneTwoFunc struct {};

func (*oneTwoFunc) NewFrame() *Frame {
	return &Frame{nil, &[2]Value {}};
}

func (*oneTwoFunc) Call(fr *Frame) {
	fr.Vars[0].(IntValue).Set(1);
	fr.Vars[1].(IntValue).Set(2);
}

type voidFunc struct {};

func (*voidFunc) NewFrame() *Frame {
	return &Frame{nil, []Value {}};
}

func (*voidFunc) Call(fr *Frame) {
}

func newTestScope() *Scope {
	sc := universe.ChildScope();
	p := token.Position{"<testScope>", 0, 0, 0};

	def := func(name string, t Type, val interface{}) {
		v, _ := sc.DefineVar(name, p, t);
		v.Init = toValue(val);
	};

	sc.DefineConst("c", p, IdealIntType, toValue(bignum.Int(1)));
	def("i", IntType, 1);
	def("i2", IntType, 2);
	def("u", UintType, uint(1));
	def("f", FloatType, 1.0);
	def("s", StringType, "abc");
	def("t", NewStructType([]StructField {StructField{"a", IntType, false}}), vstruct{1});
	def("ai", NewArrayType(2, IntType), varray{1, 2});
	def("aai", NewArrayType(2, NewArrayType(2, IntType)), varray{varray{1,2}, varray{3,4}});
	def("aai2", NewArrayType(2, NewArrayType(2, IntType)), varray{varray{5,6}, varray{7,8}});
	def("fn", NewFuncType([]Type{IntType}, false, []Type {IntType}), &testFunc{});
	def("oneTwo", NewFuncType([]Type{}, false, []Type {IntType, IntType}), &oneTwoFunc{});
	def("void", NewFuncType([]Type{}, false, []Type {}), &voidFunc{});
	def("sli", NewSliceType(IntType), vslice{varray{1, 2, 3}, 2, 3});

	return sc;
}
