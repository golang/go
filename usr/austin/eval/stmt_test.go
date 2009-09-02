// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import "testing"

var atLeastOneDecl = "at least one new variable must be declared";

var stmtTests = []test {
	// Short declarations
	Val1("x := i", "x", 1),
	Val1("x := f", "x", 1.0),
	// Type defaulting
	Val1("a := 42", "a", 42),
	Val1("a := 1.0", "a", 1.0),
	// Parallel assignment
	Val2("a, b := 1, 2", "a", 1, "b", 2),
	Val2("a, i := 1, 2", "a", 1, "i", 2),
	SErr("a, i := 1, f", opTypes),
	// TODO(austin) The parser produces an error message for this
	// one that's inconsistent with the errors I give for other
	// things
	//SErr("a, b := 1, 2, 3", "too many"),
	SErr("a, b := 1, 2, 3", "arity"),
	SErr("a := 1, 2", "too many"),
	SErr("a, b := 1", "not enough"),
	// Mixed declarations
	SErr("i := 1", atLeastOneDecl),
	SErr("i, u := 1, 2", atLeastOneDecl),
	Val2("i, x := 2, f", "i", 2, "x", 1.0),
	// Various errors
	SErr("1 := 2", "left side of := must be a name"),
	SErr("c, a := 1, 1", "cannot assign"),
	// Unpacking
	Val2("x, y := oneTwo()", "x", 1, "y", 2),
	SErr("x := oneTwo()", "too many"),
	SErr("x, y, z := oneTwo()", "not enough"),
	SErr("x, y := oneTwo(), 2", "multi-valued"),
	SErr("x := oneTwo()+2", opTypes),
	// TOOD(austin) This error message is weird
	SErr("x := void()", "not enough"),
	// Placeholders
	SErr("x := 1+\"x\"; i=x+1", opTypes),

	// Assignment
	Val1("i = 2", "i", 2),
	Val1("(i) = 2", "i", 2),
	SErr("1 = 2", "cannot assign"),
	SErr("1-1 = 2", "- expression"),
	Val1("i = 2.0", "i", 2),
	SErr("i = 2.2", constantTruncated),
	SErr("u = -2", constantUnderflows),
	SErr("i = f", opTypes),
	SErr("i, u = 0, f", opTypes),
	SErr("i, u = 0, f", "value 2"),
	Val2("i, i2 = i2, i", "i", 2, "i2", 1),
	SErr("c = 1", "cannot assign"),

	Val1("x := &i; *x = 2", "i", 2),

	Val1("ai[0] = 42", "ai", varray{ 42, 2 }),
	Val1("aai[1] = ai; ai[0] = 42", "aai", varray{ varray{1, 2}, varray{1, 2} }),
	Val1("aai = aai2", "aai", varray{ varray{5, 6}, varray{7, 8} }),

	// Assignment conversions
	SRuns("var sl []int; sl = &ai"),
	SErr("type ST []int; type AT *[2]int; var x AT = &ai; var y ST = x", opTypes),
	SRuns("type ST []int; var y ST = &ai"),
	SRuns("type AT *[2]int; var x AT = &ai; var y []int = x"),

	// Op-assignment
	Val1("i += 2", "i", 3),
	Val1("f += 2", "f", 3.0),
	SErr("2 += 2", "cannot assign"),
	SErr("i, j += 2", "cannot be combined"),
	SErr("i += 2, 3", "cannot be combined"),
	Val2("s2 := s; s += \"def\"", "s2", "abc", "s", "abcdef"),
	SErr("s += 1", opTypes),
	// Single evaluation
	Val2("ai[func()int{i+=1;return 0}()] *= 3; i2 = ai[0]", "i", 2, "i2", 3),

	// Type declarations
	// Identifiers
	SRuns("type T int"),
	SErr("type T x", "undefined"),
	SErr("type T c", "constant"),
	SErr("type T i", "variable"),
	SErr("type T T", "recursive"),
	SErr("type T x; type U T; var v U; v = 1", "undefined"),
	// Pointer types
	SRuns("type T *int"),
	SRuns("type T *T"),
	// Array types
	SRuns("type T [5]int"),
	SRuns("type T [c+42/2]int"),
	SRuns("type T [2.0]int"),
	SErr("type T [i]int", "constant expression"),
	SErr("type T [2.5]int", constantTruncated),
	SErr("type T [-1]int", "negative"),
	SErr("type T [2]T", "recursive"),
	// Struct types
	SRuns("type T struct { a int; b int }"),
	SRuns("type T struct { a int; int }"),
	SRuns("type T struct { x *T }"),
	SRuns("type T int; type U struct { T }"),
	SErr("type T *int; type U struct { T }", "embedded.*pointer"),
	SErr("type T *struct { T }", "embedded.*pointer"),
	SErr("type T struct { a int; a int }", " a .*redeclared.*:1:17"),
	SErr("type T struct { int; int }", "int .*redeclared.*:1:17"),
	SErr("type T struct { int int; int }", "int .*redeclared.*:1:17"),
	SRuns("type T struct { x *struct { T } }"),
	SErr("type T struct { x struct { T } }", "recursive"),
	SErr("type T struct { x }; type U struct { T }", "undefined"),
	// Function types
	SRuns("type T func()"),
	SRuns("type T func(a, b int) int"),
	SRuns("type T func(a, b int) (x int, y int)"),
	SRuns("type T func(a, a int) (a int, a int)"),
	SRuns("type T func(a, b int) (x, y int)"),
	SRuns("type T func(int, int) (int, int)"),
	SErr("type T func(x); type U T", "undefined"),
	SErr("type T func(a T)", "recursive"),
	// Parens
	SRuns("type T (int)"),

	// Variable declarations
	Val2("var x int", "i", 1, "x", 0),
	Val1("var x = 1", "x", 1),
	Val1("var x = 1.0", "x", 1.0),
	Val1("var x int = 1.0", "x", 1),
	// Placeholders
	SErr("var x foo; x = 1", "undefined"),
	SErr("var x foo = 1; x = 1", "undefined"),
	// Redeclaration
	SErr("var i, x int", " i .*redeclared"),
	SErr("var x int; var x int", " x .*redeclared.*:1:5"),

	// Expression statements
	SErr("1-1", "expression statement"),
	SErr("1-1", "- expression"),
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
	SErr("s++", opTypes),
	SErr("s++", "'\\+\\+'"),
	SErr("2++", "cannot assign"),
	SErr("c++", "cannot assign"),

	// Function scoping
	Val1("fn1 := func() { i=2 }; fn1()", "i", 2),
	Val1("fn1 := func() { i:=2 }; fn1()", "i", 1),
	Val2("fn1 := func() int { i=2; i:=3; i=4; return i }; x := fn1()", "i", 2, "x", 4),

	// Basic returns
	SErr("fn1 := func() int {}", "return"),
	SRuns("fn1 := func() {}"),
	SErr("fn1 := func() (r int) {}", "return"),
	Val1("fn1 := func() (r int) {return}; i = fn1()", "i", 0),
	Val1("fn1 := func() (r int) {r = 2; return}; i = fn1()", "i", 2),
	Val1("fn1 := func() (r int) {return 2}; i = fn1()", "i", 2),
	Val1("fn1 := func(int) int {return 2}; i = fn1(1)", "i", 2),

	// Multi-valued returns
	Val2("fn1 := func() (bool, int) {return true, 2}; x, y := fn1()", "x", true, "y", 2),
	SErr("fn1 := func() int {return}", "not enough values"),
	SErr("fn1 := func() int {return 1,2}", "too many values"),
	SErr("fn1 := func() {return 1}", "too many values"),
	SErr("fn1 := func() (int,int,int) {return 1,2}", "not enough values"),
	Val2("fn1 := func() (int, int) {return oneTwo()}; x, y := fn1()", "x", 1, "y", 2),
	SErr("fn1 := func() int {return oneTwo()}", "too many values"),
	SErr("fn1 := func() (int,int,int) {return oneTwo()}", "not enough values"),
	Val1("fn1 := func(x,y int) int {return x+y}; x := fn1(oneTwo())", "x", 3),

	// Return control flow
	Val2("fn1 := func(x *int) bool { *x = 2; return true; *x = 3; }; x := fn1(&i)", "i", 2, "x", true),

	// Break/continue/goto/fallthrough
	SErr("break", "outside"),
	SErr("break foo", "break.*foo.*not defined"),
	SErr("continue", "outside"),
	SErr("continue foo", "continue.*foo.*not defined"),
	SErr("fallthrough", "outside"),
	SErr("goto foo", "foo.*not defined"),
	SErr(" foo: foo:;", "foo.*redeclared.*:1:2"),
	Val1("i+=2; goto L; i+=4; L: i+=8", "i", 1+2+8),
	// Return checking
	SErr("fn1 := func() int { goto L; return 1; L: }", "return"),
	SRuns("fn1 := func() int { L: goto L; i = 2 }"),
	SRuns("fn1 := func() int { return 1; L: goto L }"),
	// Scope checking
	SRuns("fn1 := func() { { L: x:=1 } goto L }"),
	SErr("fn1 := func() { { x:=1; L: } goto L }", "into scope"),
	SErr("fn1 := func() { goto L; x:=1; L: }", "into scope"),
	SRuns("fn1 := func() { goto L; { L: x:=1 } }"),
	SErr("fn1 := func() { goto L; { x:=1; L: } }", "into scope"),

	// Blocks
	SErr("fn1 := func() int {{}}", "return"),
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
	SErr("if true { x := 2 }; x = 4", undefined),
	Val2("if i := 2; true { i2 = i; i := 3 }", "i", 1, "i2", 2),
	Val2("if i := 2; false {} else { i2 = i; i := 3 }", "i", 1, "i2", 2),
	// Return checking
	SRuns("fn1 := func() int { if true { return 1 } else { return 2 } }"),
	SRuns("fn1 := func() int { if true { return 1 } else return 2 }"),
	SErr("fn1 := func() int { if true { return 1 } else { } }", "return"),
	SErr("fn1 := func() int { if true { } else { return 1 } }", "return"),
	SErr("fn1 := func() int { if true { } else return 1 }", "return"),
	SErr("fn1 := func() int { if true { } else { } }", "return"),
	SErr("fn1 := func() int { if true { return 1 } }", "return"),
	SErr("fn1 := func() int { if true { } }", "return"),
	SRuns("fn1 := func() int { if true { }; return 1 }"),
	SErr("fn1 := func() int { if { } }", "return"),
	SErr("fn1 := func() int { if { } else { return 2 } }", "return"),
	SRuns("fn1 := func() int { if { return 1 } }"),
	SRuns("fn1 := func() int { if { return 1 } else { } }"),
	SRuns("fn1 := func() int { if { return 1 } else { } }"),

	// Switch
	Val1("switch { case false: i += 2; case true: i += 4; default: i += 8 }", "i", 1+4),
	Val1("switch { default: i += 2; case false: i += 4; case true: i += 8 }", "i", 1+8),
	SErr("switch { default: i += 2; default: i += 4 }", "more than one"),
	Val1("switch false { case false: i += 2; case true: i += 4; default: i += 8 }", "i", 1+2),
	SErr("switch s { case 1: }", opTypes),
	SErr("switch ai { case ai: i += 2 }", opTypes),
	Val1("switch 1.0 { case 1: i += 2; case 2: i += 4 }", "i", 1+2),
	Val1("switch 1.5 { case 1: i += 2; case 2: i += 4 }", "i", 1),
	SErr("switch oneTwo() {}", "multi-valued expression"),
	Val1("switch 2 { case 1: i += 2; fallthrough; case 2: i += 4; fallthrough; case 3: i += 8; fallthrough }", "i", 1+4+8),
	Val1("switch 5 { case 1: i += 2; fallthrough; default: i += 4; fallthrough; case 2: i += 8; fallthrough; case 3: i += 16; fallthrough }", "i", 1+4+8+16),
	SErr("switch { case true: fallthrough; i += 2 }", "final statement"),
	Val1("switch { case true: i += 2; fallthrough; ; ; case false: i += 4 }", "i", 1+2+4),
	Val1("switch 2 { case 0, 1: i += 2; case 2, 3: i += 4 }", "i", 1+4),
	Val2("switch func()int{i2++;return 5}() { case 1, 2: i += 2; case 4, 5: i += 4 }", "i", 1+4, "i2", 3),
	SRuns("switch i { case i: }"),
	// TODO(austin) Why doesn't this fail?
	SErr("case 1:", "XXX"),

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
	Val1("L1: for { L2: for { i+=2; break L1; i+=4 } i+=8 }", "i", 1+2),
	Val1("L1: for { L2: for { i+=2; break L2; i+=4 } i+=8; break; i+=16 }", "i", 1+2+8),
	SErr("L1: { for { break L1 } }", "break.*not defined"),
	SErr("L1: for {} for { break L1 }", "break.*not defined"),
	SErr("L1:; for { break L1 }", "break.*not defined"),
	Val2("L1: for i = 0; i < 2; i++ { L2: for { i2++; continue L1; i2++ } }", "i", 2, "i2", 4),
	SErr("L1: { for { continue L1 } }", "continue.*not defined"),
	SErr("L1:; for { continue L1 }", "continue.*not defined"),
	// Return checking
	SRuns("fn1 := func() int{ for {} }"),
	SErr("fn1 := func() int{ for true {} }", "return"),
	SErr("fn1 := func() int{ for true {return 1} }", "return"),
	SErr("fn1 := func() int{ for {break} }", "return"),
	SRuns("fn1 := func() int{ for { for {break} } }"),
	SErr("fn1 := func() int{ L1: for { for {break L1} } }", "return"),
	SRuns("fn1 := func() int{ for true {} return 1 }"),

	// Selectors
	Val1("var x struct { a int; b int }; x.a = 42; i = x.a", "i", 42),
	Val1("type T struct { x int }; var y struct { T }; y.x = 42; i = y.x", "i", 42),
	Val2("type T struct { x int }; var y struct { T; x int }; y.x = 42; i = y.x; i2 = y.T.x", "i", 42, "i2", 0),
	SRuns("type T struct { x int }; var y struct { *T }; a := func(){i=y.x}"),
	SErr("type T struct { x int }; var x T; x.y = 42", "no field"),
	SErr("type T struct { x int }; type U struct { x int }; var y struct { T; U }; y.x = 42", "ambiguous.*\tT\\.x\n\tU\\.x"),
	SErr("type T struct { *T }; var x T; x.foo", "no field"),

	//Val1("fib := func(int) int{return 0;}; fib = func(v int) int { if v < 2 { return 1 } return fib(v-1)+fib(v-2) }; i = fib(20)", "i", 0),

	// Make slice
	Val2("x := make([]int, 2); x[0] = 42; i, i2 = x[0], x[1]", "i", 42, "i2", 0),
	Val2("x := make([]int, 2); x[1] = 42; i, i2 = x[0], x[1]", "i", 0, "i2", 42),
	SRTErr("x := make([]int, 2); x[-i] = 42", "negative index"),
	SRTErr("x := make([]int, 2); x[2] = 42", "index 2 exceeds"),
	Val2("x := make([]int, 2, 3); i, i2 = len(x), cap(x)", "i", 2, "i2", 3),
	Val2("x := make([]int, 3, 2); i, i2 = len(x), cap(x)", "i", 3, "i2", 3),
	SRTErr("x := make([]int, -i)", "negative length"),
	SRTErr("x := make([]int, 2, -i)", "negative capacity"),
	SRTErr("x := make([]int, 2, 3); x[2] = 42", "index 2 exceeds"),
	SErr("x := make([]int, 2, 3, 4)", "too many"),
	SErr("x := make([]int)", "not enough"),

	// TODO(austin) Test make map

	// Maps
	Val1("x := make(map[int] int); x[1] = 42; i = x[1]", "i", 42),
	Val2("x := make(map[int] int); x[1] = 42; i, y := x[1]", "i", 42, "y", true),
	Val2("x := make(map[int] int); x[1] = 42; i, y := x[2]", "i", 0, "y", false),
	// Not implemented
	//Val1("x := make(map[int] int); x[1] = 42, true; i = x[1]", "i", 42),
	//Val2("x := make(map[int] int); x[1] = 42; x[1] = 42, false; i, y := x[1]", "i", 0, "y", false),
	SRuns("var x int; a := make(map[int] int); a[0], x = 1, 2"),
	SErr("x := make(map[int] int); (func(a,b int){})(x[0])", "not enough"),
	SErr("x := make(map[int] int); x[1] = oneTwo()", "too many"),
	SRTErr("x := make(map[int] int); i = x[1]", "key '1' not found"),
}

func TestStmt(t *testing.T) {
	runTests(t, "stmtTests", stmtTests);
}
