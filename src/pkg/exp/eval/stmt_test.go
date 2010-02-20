// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import "testing"

var atLeastOneDecl = "at least one new variable must be declared"

var stmtTests = []test{
	// Short declarations
	Val1("x := i", "x", 1),
	Val1("x := f", "x", 1.0),
	// Type defaulting
	Val1("a := 42", "a", 42),
	Val1("a := 1.0", "a", 1.0),
	// Parallel assignment
	Val2("a, b := 1, 2", "a", 1, "b", 2),
	Val2("a, i := 1, 2", "a", 1, "i", 2),
	CErr("a, i := 1, f", opTypes),
	CErr("a, b := 1, 2, 3", "too many"),
	CErr("a := 1, 2", "too many"),
	CErr("a, b := 1", "not enough"),
	// Mixed declarations
	CErr("i := 1", atLeastOneDecl),
	CErr("i, u := 1, 2", atLeastOneDecl),
	Val2("i, x := 2, f", "i", 2, "x", 1.0),
	// Various errors
	CErr("1 := 2", "left side of := must be a name"),
	CErr("c, a := 1, 1", "cannot assign"),
	// Unpacking
	Val2("x, y := oneTwo()", "x", 1, "y", 2),
	CErr("x := oneTwo()", "too many"),
	CErr("x, y, z := oneTwo()", "not enough"),
	CErr("x, y := oneTwo(), 2", "multi-valued"),
	CErr("x := oneTwo()+2", opTypes),
	// TOOD(austin) This error message is weird
	CErr("x := void()", "not enough"),
	// Placeholders
	CErr("x := 1+\"x\"; i=x+1", opTypes),

	// Assignment
	Val1("i = 2", "i", 2),
	Val1("(i) = 2", "i", 2),
	CErr("1 = 2", "cannot assign"),
	CErr("1-1 = 2", "- expression"),
	Val1("i = 2.0", "i", 2),
	CErr("i = 2.2", constantTruncated),
	CErr("u = -2", constantUnderflows),
	CErr("i = f", opTypes),
	CErr("i, u = 0, f", opTypes),
	CErr("i, u = 0, f", "value 2"),
	Val2("i, i2 = i2, i", "i", 2, "i2", 1),
	CErr("c = 1", "cannot assign"),

	Val1("x := &i; *x = 2", "i", 2),

	Val1("ai[0] = 42", "ai", varray{42, 2}),
	Val1("aai[1] = ai; ai[0] = 42", "aai", varray{varray{1, 2}, varray{1, 2}}),
	Val1("aai = aai2", "aai", varray{varray{5, 6}, varray{7, 8}}),

	// Assignment conversions
	Run("var sl []int; sl = &ai"),
	CErr("type ST []int; type AT *[2]int; var x AT = &ai; var y ST = x", opTypes),
	Run("type ST []int; var y ST = &ai"),
	Run("type AT *[2]int; var x AT = &ai; var y []int = x"),

	// Op-assignment
	Val1("i += 2", "i", 3),
	Val("i", 1),
	Val1("f += 2", "f", 3.0),
	CErr("2 += 2", "cannot assign"),
	CErr("i, j += 2", "cannot be combined"),
	CErr("i += 2, 3", "cannot be combined"),
	Val2("s2 := s; s += \"def\"", "s2", "abc", "s", "abcdef"),
	CErr("s += 1", opTypes),
	// Single evaluation
	Val2("ai[func()int{i+=1;return 0}()] *= 3; i2 = ai[0]", "i", 2, "i2", 3),

	// Type declarations
	// Identifiers
	Run("type T int"),
	CErr("type T x", "undefined"),
	CErr("type T c", "constant"),
	CErr("type T i", "variable"),
	CErr("type T T", "recursive"),
	CErr("type T x; type U T; var v U; v = 1", "undefined"),
	// Pointer types
	Run("type T *int"),
	Run("type T *T"),
	// Array types
	Run("type T [5]int"),
	Run("type T [c+42/2]int"),
	Run("type T [2.0]int"),
	CErr("type T [i]int", "constant expression"),
	CErr("type T [2.5]int", constantTruncated),
	CErr("type T [-1]int", "negative"),
	CErr("type T [2]T", "recursive"),
	// Struct types
	Run("type T struct { a int; b int }"),
	Run("type T struct { a int; int }"),
	Run("type T struct { x *T }"),
	Run("type T int; type U struct { T }"),
	CErr("type T *int; type U struct { T }", "embedded.*pointer"),
	CErr("type T *struct { T }", "embedded.*pointer"),
	CErr("type T struct { a int; a int }", " a .*redeclared.*:1:17"),
	CErr("type T struct { int; int }", "int .*redeclared.*:1:17"),
	CErr("type T struct { int int; int }", "int .*redeclared.*:1:17"),
	Run("type T struct { x *struct { T } }"),
	CErr("type T struct { x struct { T } }", "recursive"),
	CErr("type T struct { x }; type U struct { T }", "undefined"),
	// Function types
	Run("type T func()"),
	Run("type T func(a, b int) int"),
	Run("type T func(a, b int) (x int, y int)"),
	Run("type T func(a, a int) (a int, a int)"),
	Run("type T func(a, b int) (x, y int)"),
	Run("type T func(int, int) (int, int)"),
	CErr("type T func(x); type U T", "undefined"),
	CErr("type T func(a T)", "recursive"),
	// Interface types
	Run("type T interface {x(a, b int) int}"),
	Run("type T interface {x(a, b int) int}; type U interface {T; y(c int)}"),
	CErr("type T interface {x(a int); x()}", "method x redeclared"),
	CErr("type T interface {x()}; type U interface {T; x()}", "method x redeclared"),
	CErr("type T int; type U interface {T}", "embedded type"),
	// Parens
	Run("type T (int)"),

	// Variable declarations
	Val2("var x int", "i", 1, "x", 0),
	Val1("var x = 1", "x", 1),
	Val1("var x = 1.0", "x", 1.0),
	Val1("var x int = 1.0", "x", 1),
	// Placeholders
	CErr("var x foo; x = 1", "undefined"),
	CErr("var x foo = 1; x = 1", "undefined"),
	// Redeclaration
	CErr("var i, x int", " i .*redeclared"),
	CErr("var x int; var x int", " x .*redeclared.*:1:5"),

	// Expression statements
	CErr("x := func(){ 1-1 }", "expression statement"),
	CErr("x := func(){ 1-1 }", "- expression"),
	Val1("fn(2)", "i", 1),

	// IncDec statements
	Val1("i++", "i", 2),
	Val1("i--", "i", 0),
	Val1("u++", "u", uint(2)),
	Val1("u--", "u", uint(0)),
	Val1("f++", "f", 2.0),
	Val1("f--", "f", 0.0),
	// Single evaluation
	Val2("ai[func()int{i+=1;return 0}()]++; i2 = ai[0]", "i", 2, "i2", 2),
	// Operand types
	CErr("s++", opTypes),
	CErr("s++", "'\\+\\+'"),
	CErr("2++", "cannot assign"),
	CErr("c++", "cannot assign"),

	// Function scoping
	Val1("fn1 := func() { i=2 }; fn1()", "i", 2),
	Val1("fn1 := func() { i:=2 }; fn1()", "i", 1),
	Val2("fn1 := func() int { i=2; i:=3; i=4; return i }; x := fn1()", "i", 2, "x", 4),

	// Basic returns
	CErr("fn1 := func() int {}", "return"),
	Run("fn1 := func() {}"),
	CErr("fn1 := func() (r int) {}", "return"),
	Val1("fn1 := func() (r int) {return}; i = fn1()", "i", 0),
	Val1("fn1 := func() (r int) {r = 2; return}; i = fn1()", "i", 2),
	Val1("fn1 := func() (r int) {return 2}; i = fn1()", "i", 2),
	Val1("fn1 := func(int) int {return 2}; i = fn1(1)", "i", 2),

	// Multi-valued returns
	Val2("fn1 := func() (bool, int) {return true, 2}; x, y := fn1()", "x", true, "y", 2),
	CErr("fn1 := func() int {return}", "not enough values"),
	CErr("fn1 := func() int {return 1,2}", "too many values"),
	CErr("fn1 := func() {return 1}", "too many values"),
	CErr("fn1 := func() (int,int,int) {return 1,2}", "not enough values"),
	Val2("fn1 := func() (int, int) {return oneTwo()}; x, y := fn1()", "x", 1, "y", 2),
	CErr("fn1 := func() int {return oneTwo()}", "too many values"),
	CErr("fn1 := func() (int,int,int) {return oneTwo()}", "not enough values"),
	Val1("fn1 := func(x,y int) int {return x+y}; x := fn1(oneTwo())", "x", 3),

	// Return control flow
	Val2("fn1 := func(x *int) bool { *x = 2; return true; *x = 3; }; x := fn1(&i)", "i", 2, "x", true),

	// Break/continue/goto/fallthrough
	CErr("break", "outside"),
	CErr("break foo", "break.*foo.*not defined"),
	CErr("continue", "outside"),
	CErr("continue foo", "continue.*foo.*not defined"),
	CErr("fallthrough", "outside"),
	CErr("goto foo", "foo.*not defined"),
	CErr(" foo: foo:;", "foo.*redeclared.*:1:2"),
	Val1("i+=2; goto L; i+=4; L: i+=8", "i", 1+2+8),
	// Return checking
	CErr("fn1 := func() int { goto L; return 1; L: }", "return"),
	Run("fn1 := func() int { L: goto L; i = 2 }"),
	Run("fn1 := func() int { return 1; L: goto L }"),
	// Scope checking
	Run("fn1 := func() { { L: x:=1 }; goto L }"),
	CErr("fn1 := func() { { x:=1; L: }; goto L }", "into scope"),
	CErr("fn1 := func() { goto L; x:=1; L: }", "into scope"),
	Run("fn1 := func() { goto L; { L: x:=1 } }"),
	CErr("fn1 := func() { goto L; { x:=1; L: } }", "into scope"),

	// Blocks
	CErr("fn1 := func() int {{}}", "return"),
	Val1("fn1 := func() bool { { return true } }; b := fn1()", "b", true),

	// If
	Val2("if true { i = 2 } else { i = 3 }; i2 = 4", "i", 2, "i2", 4),
	Val2("if false { i = 2 } else { i = 3 }; i2 = 4", "i", 3, "i2", 4),
	Val2("if i == i2 { i = 2 } else { i = 3 }; i2 = 4", "i", 3, "i2", 4),
	// Omit optional parts
	Val2("if { i = 2 } else { i = 3 }; i2 = 4", "i", 2, "i2", 4),
	Val2("if true { i = 2 }; i2 = 4", "i", 2, "i2", 4),
	Val2("if false { i = 2 }; i2 = 4", "i", 1, "i2", 4),
	// Init
	Val2("if x := true; x { i = 2 } else { i = 3 }; i2 = 4", "i", 2, "i2", 4),
	Val2("if x := false; x { i = 2 } else { i = 3 }; i2 = 4", "i", 3, "i2", 4),
	// Statement else
	Val2("if true { i = 2 } else i = 3; i2 = 4", "i", 2, "i2", 4),
	Val2("if false { i = 2 } else i = 3; i2 = 4", "i", 3, "i2", 4),
	// Scoping
	Val2("if true { i := 2 } else { i := 3 }; i2 = i", "i", 1, "i2", 1),
	Val2("if false { i := 2 } else { i := 3 }; i2 = i", "i", 1, "i2", 1),
	Val2("if false { i := 2 } else i := 3; i2 = i", "i", 1, "i2", 1),
	CErr("if true { x := 2 }; x = 4", undefined),
	Val2("if i := 2; true { i2 = i; i := 3 }", "i", 1, "i2", 2),
	Val2("if i := 2; false {} else { i2 = i; i := 3 }", "i", 1, "i2", 2),
	// Return checking
	Run("fn1 := func() int { if true { return 1 } else { return 2 } }"),
	Run("fn1 := func() int { if true { return 1 } else return 2 }"),
	CErr("fn1 := func() int { if true { return 1 } else { } }", "return"),
	CErr("fn1 := func() int { if true { } else { return 1 } }", "return"),
	CErr("fn1 := func() int { if true { } else return 1 }", "return"),
	CErr("fn1 := func() int { if true { } else { } }", "return"),
	CErr("fn1 := func() int { if true { return 1 } }", "return"),
	CErr("fn1 := func() int { if true { } }", "return"),
	Run("fn1 := func() int { if true { }; return 1 }"),
	CErr("fn1 := func() int { if { } }", "return"),
	CErr("fn1 := func() int { if { } else { return 2 } }", "return"),
	Run("fn1 := func() int { if { return 1 } }"),
	Run("fn1 := func() int { if { return 1 } else { } }"),
	Run("fn1 := func() int { if { return 1 } else { } }"),

	// Switch
	Val1("switch { case false: i += 2; case true: i += 4; default: i += 8 }", "i", 1+4),
	Val1("switch { default: i += 2; case false: i += 4; case true: i += 8 }", "i", 1+8),
	CErr("switch { default: i += 2; default: i += 4 }", "more than one"),
	Val1("switch false { case false: i += 2; case true: i += 4; default: i += 8 }", "i", 1+2),
	CErr("switch s { case 1: }", opTypes),
	CErr("switch ai { case ai: i += 2 }", opTypes),
	Val1("switch 1.0 { case 1: i += 2; case 2: i += 4 }", "i", 1+2),
	Val1("switch 1.5 { case 1: i += 2; case 2: i += 4 }", "i", 1),
	CErr("switch oneTwo() {}", "multi-valued expression"),
	Val1("switch 2 { case 1: i += 2; fallthrough; case 2: i += 4; fallthrough; case 3: i += 8; fallthrough }", "i", 1+4+8),
	Val1("switch 5 { case 1: i += 2; fallthrough; default: i += 4; fallthrough; case 2: i += 8; fallthrough; case 3: i += 16; fallthrough }", "i", 1+4+8+16),
	CErr("switch { case true: fallthrough; i += 2 }", "final statement"),
	Val1("switch { case true: i += 2; fallthrough; ; ; case false: i += 4 }", "i", 1+2+4),
	Val1("switch 2 { case 0, 1: i += 2; case 2, 3: i += 4 }", "i", 1+4),
	Val2("switch func()int{i2++;return 5}() { case 1, 2: i += 2; case 4, 5: i += 4 }", "i", 1+4, "i2", 3),
	Run("switch i { case i: }"),
	// TODO(austin) Why doesn't this fail?
	//CErr("case 1:", "XXX"),

	// For
	Val2("for x := 1; x < 5; x++ { i+=x }; i2 = 4", "i", 11, "i2", 4),
	Val2("for x := 1; x < 5; x++ { i+=x; break; i++ }; i2 = 4", "i", 2, "i2", 4),
	Val2("for x := 1; x < 5; x++ { i+=x; continue; i++ }; i2 = 4", "i", 11, "i2", 4),
	Val2("for i = 2; false; i = 3 { i = 4 }; i2 = 4", "i", 2, "i2", 4),
	Val2("for i < 5 { i++ }; i2 = 4", "i", 5, "i2", 4),
	Val2("for i < 0 { i++ }; i2 = 4", "i", 1, "i2", 4),
	// Scoping
	Val2("for i := 2; true; { i2 = i; i := 3; break }", "i", 1, "i2", 2),
	// Labeled break/continue
	Val1("L1: for { L2: for { i+=2; break L1; i+=4 }; i+=8 }", "i", 1+2),
	Val1("L1: for { L2: for { i+=2; break L2; i+=4 }; i+=8; break; i+=16 }", "i", 1+2+8),
	CErr("L1: { for { break L1 } }", "break.*not defined"),
	CErr("L1: for {}; for { break L1 }", "break.*not defined"),
	CErr("L1:; for { break L1 }", "break.*not defined"),
	Val2("L1: for i = 0; i < 2; i++ { L2: for { i2++; continue L1; i2++ } }", "i", 2, "i2", 4),
	CErr("L1: { for { continue L1 } }", "continue.*not defined"),
	CErr("L1:; for { continue L1 }", "continue.*not defined"),
	// Return checking
	Run("fn1 := func() int{ for {} }"),
	CErr("fn1 := func() int{ for true {} }", "return"),
	CErr("fn1 := func() int{ for true {return 1} }", "return"),
	CErr("fn1 := func() int{ for {break} }", "return"),
	Run("fn1 := func() int{ for { for {break} } }"),
	CErr("fn1 := func() int{ L1: for { for {break L1} } }", "return"),
	Run("fn1 := func() int{ for true {}; return 1 }"),

	// Selectors
	Val1("var x struct { a int; b int }; x.a = 42; i = x.a", "i", 42),
	Val1("type T struct { x int }; var y struct { T }; y.x = 42; i = y.x", "i", 42),
	Val2("type T struct { x int }; var y struct { T; x int }; y.x = 42; i = y.x; i2 = y.T.x", "i", 42, "i2", 0),
	Run("type T struct { x int }; var y struct { *T }; a := func(){i=y.x}"),
	CErr("type T struct { x int }; var x T; x.y = 42", "no field"),
	CErr("type T struct { x int }; type U struct { x int }; var y struct { T; U }; y.x = 42", "ambiguous.*\tT\\.x\n\tU\\.x"),
	CErr("type T struct { *T }; var x T; x.foo", "no field"),

	Val1("fib := func(int) int{return 0;}; fib = func(v int) int { if v < 2 { return 1 }; return fib(v-1)+fib(v-2) }; i = fib(20)", "i", 10946),

	// Make slice
	Val2("x := make([]int, 2); x[0] = 42; i, i2 = x[0], x[1]", "i", 42, "i2", 0),
	Val2("x := make([]int, 2); x[1] = 42; i, i2 = x[0], x[1]", "i", 0, "i2", 42),
	RErr("x := make([]int, 2); x[-i] = 42", "negative index"),
	RErr("x := make([]int, 2); x[2] = 42", "index 2 exceeds"),
	Val2("x := make([]int, 2, 3); i, i2 = len(x), cap(x)", "i", 2, "i2", 3),
	Val2("x := make([]int, 3, 2); i, i2 = len(x), cap(x)", "i", 3, "i2", 3),
	RErr("x := make([]int, -i)", "negative length"),
	RErr("x := make([]int, 2, -i)", "negative capacity"),
	RErr("x := make([]int, 2, 3); x[2] = 42", "index 2 exceeds"),
	CErr("x := make([]int, 2, 3, 4)", "too many"),
	CErr("x := make([]int)", "not enough"),

	// TODO(austin) Test make map

	// Maps
	Val1("x := make(map[int] int); x[1] = 42; i = x[1]", "i", 42),
	Val2("x := make(map[int] int); x[1] = 42; i, y := x[1]", "i", 42, "y", true),
	Val2("x := make(map[int] int); x[1] = 42; i, y := x[2]", "i", 0, "y", false),
	// Not implemented
	//Val1("x := make(map[int] int); x[1] = 42, true; i = x[1]", "i", 42),
	//Val2("x := make(map[int] int); x[1] = 42; x[1] = 42, false; i, y := x[1]", "i", 0, "y", false),
	Run("var x int; a := make(map[int] int); a[0], x = 1, 2"),
	CErr("x := make(map[int] int); (func(a,b int){})(x[0])", "not enough"),
	CErr("x := make(map[int] int); x[1] = oneTwo()", "too many"),
	RErr("x := make(map[int] int); i = x[1]", "key '1' not found"),

	// Functions
	Val2("func fib(n int) int { if n <= 2 { return n }; return fib(n-1) + fib(n-2) }", "fib(4)", 5, "fib(10)", 89),
	Run("func f1(){}"),
	Run2("func f1(){}", "f1()"),
}

func TestStmt(t *testing.T) { runTests(t, "stmtTests", stmtTests) }
