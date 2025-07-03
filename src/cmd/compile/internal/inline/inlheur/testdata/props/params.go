// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>
package params

import "os"

// params.go T_feeds_if_simple 20 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_simple(x int) {
	if x < 100 {
		os.Exit(1)
	}
	println(x)
}

// params.go T_feeds_if_nested 35 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
//   1 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64,32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_nested(x, y int) {
	if y != 0 {
		if x < 100 {
			os.Exit(1)
		}
	}
	println(x)
}

// params.go T_feeds_if_pointer 51 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_pointer(xp *int) {
	if xp != nil {
		os.Exit(1)
	}
	println(xp)
}

// params.go T.T_feeds_if_simple_method 66 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
//   1 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32,32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func (r T) T_feeds_if_simple_method(x int) {
	if x < 100 {
		os.Exit(1)
	}
	if r != 99 {
		os.Exit(2)
	}
	println(x)
}

// params.go T_feeds_if_blanks 86 0 1
// ParamFlags
//   0 ParamNoInfo
//   1 ParamFeedsIfOrSwitch
//   2 ParamNoInfo
//   3 ParamNoInfo
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,32,0,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_blanks(_ string, x int, _ bool, _ bool) {
	// blanks ignored; from a props perspective "x" is param 0
	if x < 100 {
		os.Exit(1)
	}
	println(x)
}

// params.go T_feeds_if_with_copy 101 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_with_copy(x int) {
	// simple copy here -- we get this case
	xx := x
	if xx < 100 {
		os.Exit(1)
	}
	println(x)
}

// params.go T_feeds_if_with_copy_expr 115 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_with_copy_expr(x int) {
	// this case (copy of expression) currently not handled.
	xx := x < 100
	if xx {
		os.Exit(1)
	}
	println(x)
}

// params.go T_feeds_switch 131 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_switch(x int) {
	switch x {
	case 101:
		println(101)
	case 202:
		panic("bad")
	}
	println(x)
}

// params.go T_feeds_if_toocomplex 146 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_toocomplex(x int, y int) {
	// not handled at the moment; we only look for cases where
	// an "if" or "switch" can be simplified based on a single
	// constant param, not a combination of constant params.
	if x < y {
		panic("bad")
	}
	println(x + y)
}

// params.go T_feeds_if_redefined 161 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_redefined(x int) {
	if x < G {
		x++
	}
	if x == 101 {
		panic("bad")
	}
}

// params.go T_feeds_if_redefined2 175 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_redefined2(x int) {
	// this currently classifies "x" as "no info", since the analysis we
	// use to check for reassignments/redefinitions is not flow-sensitive,
	// but we could probably catch this case with better analysis or
	// high-level SSA.
	if x == 101 {
		panic("bad")
	}
	if x < G {
		x++
	}
}

// params.go T_feeds_multi_if 196 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
//   1 ParamNoInfo
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_multi_if(x int, y int) {
	// Here we have one "if" that is too complex (x < y) but one that is
	// simple enough. Currently we enable the heuristic for this. It's
	// possible to imagine this being a bad thing if the function in
	// question is sufficiently large, but if it's too large we probably
	// can't inline it anyhow.
	if x < y {
		panic("bad")
	}
	if x < 10 {
		panic("whatev")
	}
	println(x + y)
}

// params.go T_feeds_if_redefined_indirectwrite 216 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_redefined_indirectwrite(x int) {
	ax := &x
	if G != 2 {
		*ax = G
	}
	if x == 101 {
		panic("bad")
	}
}

// params.go T_feeds_if_redefined_indirectwrite_copy 231 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_redefined_indirectwrite_copy(x int) {
	// we don't catch this case, "x" is marked as no info,
	// since we're conservative about redefinitions.
	ax := &x
	cx := x
	if G != 2 {
		*ax = G
	}
	if cx == 101 {
		panic("bad")
	}
}

// params.go T_feeds_if_expr1 251 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_expr1(x int) {
	if x == 101 || x == 102 || x&0xf == 0 {
		panic("bad")
	}
}

// params.go T_feeds_if_expr2 262 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_expr2(x int) {
	if (x*x)-(x+x)%x == 101 || x&0xf == 0 {
		panic("bad")
	}
}

// params.go T_feeds_if_expr3 273 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_expr3(x int) {
	if x-(x&0x1)^378 > (1 - G) {
		panic("bad")
	}
}

// params.go T_feeds_if_shift_may_panic 284 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_shift_may_panic(x int) *int {
	// here if "x" is a constant like 2, we could simplify the "if",
	// but if we were to pass in a negative value for "x" we can't
	// fold the condition due to the need to panic on negative shift.
	if 1<<x > 1024 {
		return nil
	}
	return &G
}

// params.go T_feeds_if_maybe_divide_by_zero 299 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_maybe_divide_by_zero(x int) {
	if 99/x == 3 {
		return
	}
	println("blarg")
}

// params.go T_feeds_indcall 313 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[16],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_indcall(x func()) {
	if G != 20 {
		x()
	}
}

// params.go T_feeds_indcall_and_if 326 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall|ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[48],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_indcall_and_if(x func()) {
	if x != nil {
		x()
	}
}

// params.go T_feeds_indcall_with_copy 339 0 1
// ParamFlags
//   0 ParamFeedsIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[8],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_indcall_with_copy(x func()) {
	xx := x
	if G < 10 {
		G--
	}
	xx()
}

// params.go T_feeds_interface_method_call 354 0 1
// ParamFlags
//   0 ParamFeedsInterfaceMethodCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[2],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_feeds_interface_method_call(i I) {
	i.Blarg()
}

var G int

type T int

type I interface {
	Blarg()
}

func (r T) Blarg() {
}
