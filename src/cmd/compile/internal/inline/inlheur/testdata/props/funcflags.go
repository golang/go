// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>

package funcflags

import "os"

// funcflags.go T_simple 20 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":null,"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_simple() {
	panic("bad")
}

// funcflags.go T_nested 32 0 1
// Flags FuncPropNeverReturns
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":1,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_nested(x int) {
	if x < 10 {
		panic("bad")
	} else {
		panic("good")
	}
}

// funcflags.go T_block1 46 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_block1(x int) {
	panic("bad")
	if x < 10 {
		return
	}
}

// funcflags.go T_block2 60 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_block2(x int) {
	if x < 10 {
		return
	}
	panic("bad")
}

// funcflags.go T_switches1 75 0 1
// Flags FuncPropNeverReturns
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":1,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_switches1(x int) {
	switch x {
	case 1:
		panic("one")
	case 2:
		panic("two")
	}
	panic("whatev")
}

// funcflags.go T_switches1a 92 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_switches1a(x int) {
	switch x {
	case 2:
		panic("two")
	}
}

// funcflags.go T_switches2 106 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_switches2(x int) {
	switch x {
	case 1:
		panic("one")
	case 2:
		panic("two")
	default:
		return
	}
	panic("whatev")
}

// funcflags.go T_switches3 123 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_switches3(x interface{}) {
	switch x.(type) {
	case bool:
		panic("one")
	case float32:
		panic("two")
	}
}

// funcflags.go T_switches4 138 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_switches4(x int) {
	switch x {
	case 1:
		x++
		fallthrough
	case 2:
		panic("two")
		fallthrough
	default:
		panic("bad")
	}
	panic("whatev")
}

// funcflags.go T_recov 157 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_recov(x int) {
	if x := recover(); x != nil {
		panic(x)
	}
}

// funcflags.go T_forloops1 169 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_forloops1(x int) {
	for {
		panic("wokketa")
	}
}

// funcflags.go T_forloops2 180 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_forloops2(x int) {
	for {
		println("blah")
		if true {
			break
		}
		panic("warg")
	}
}

// funcflags.go T_forloops3 195 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_forloops3(x int) {
	for i := 0; i < 101; i++ {
		println("blah")
		if true {
			continue
		}
		panic("plark")
	}
	for i := range [10]int{} {
		println(i)
		panic("plark")
	}
	panic("whatev")
}

// funcflags.go T_hasgotos 215 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_hasgotos(x int, y int) {
	{
		xx := x
		panic("bad")
	lab1:
		goto lab2
	lab2:
		if false {
			goto lab1
		} else {
			goto lab4
		}
	lab4:
		if xx < y {
		lab3:
			if false {
				goto lab3
			}
		}
		println(9)
	}
}

// funcflags.go T_break_with_label 246 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
//   1 ParamNoInfo
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_break_with_label(x int, y int) {
	// presence of break with label should pessimize this func
	// (similar to goto).
	panic("bad")
lab1:
	for {
		println("blah")
		if x < 0 {
			break lab1
		}
		panic("hubba")
	}
}

// funcflags.go T_callsexit 268 0 1
// Flags FuncPropNeverReturns
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":1,"ParamFlags":[32],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_callsexit(x int) {
	if x < 0 {
		os.Exit(1)
	}
	os.Exit(2)
}

// funcflags.go T_exitinexpr 281 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// callsite: funcflags.go:286:18|0 flagstr "CallSiteOnPanicPath" flagval 2 score 102 mask 1 maskstr "panicPathAdj"
// <endcallsites>
// <endfuncpreamble>
func T_exitinexpr(x int) {
	// This function does indeed unconditionally call exit, since the
	// first thing it does is invoke exprcallsexit, however from the
	// perspective of this function, the call is not at the statement
	// level, so we'll wind up missing it.
	if exprcallsexit(x) < 0 {
		println("foo")
	}
}

// funcflags.go T_select_noreturn 297 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":[0,0,0],"ResultFlags":null}
// <endcallsites>
// <endfuncpreamble>
func T_select_noreturn(chi chan int, chf chan float32, p *int) {
	rv := 0
	select {
	case i := <-chi:
		rv = i
	case f := <-chf:
		rv = int(f)
	}
	*p = rv
	panic("bad")
}

// funcflags.go T_select_mayreturn 314 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0,0],"ResultFlags":[0]}
// <endcallsites>
// <endfuncpreamble>
func T_select_mayreturn(chi chan int, chf chan float32, p *int) int {
	rv := 0
	select {
	case i := <-chi:
		rv = i
		return i
	case f := <-chf:
		rv = int(f)
	}
	*p = rv
	panic("bad")
}

// funcflags.go T_calls_callsexit 334 0 1
// Flags FuncPropNeverReturns
// <endpropsdump>
// {"Flags":1,"ParamFlags":[0],"ResultFlags":null}
// callsite: funcflags.go:335:15|0 flagstr "CallSiteOnPanicPath" flagval 2 score 102 mask 1 maskstr "panicPathAdj"
// <endcallsites>
// <endfuncpreamble>
func T_calls_callsexit(x int) {
	exprcallsexit(x)
}

func exprcallsexit(x int) int {
	os.Exit(x)
	return x
}
