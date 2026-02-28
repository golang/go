// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>
package calls

import "os"

// calls.go T_call_in_panic_arg 19 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// callsite: calls.go:21:15|0 flagstr "CallSiteOnPanicPath" flagval 2 score 42 mask 1 maskstr "panicPathAdj"
// <endcallsites>
// <endfuncpreamble>
func T_call_in_panic_arg(x int) {
	if x < G {
		panic(callee(x))
	}
}

// calls.go T_calls_in_loops 32 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// callsite: calls.go:34:9|0 flagstr "CallSiteInLoop" flagval 1 score -3 mask 4 maskstr "inLoopAdj"
// callsite: calls.go:37:9|1 flagstr "CallSiteInLoop" flagval 1 score -3 mask 4 maskstr "inLoopAdj"
// <endcallsites>
// <endfuncpreamble>
func T_calls_in_loops(x int, q []string) {
	for i := 0; i < x; i++ {
		callee(i)
	}
	for _, s := range q {
		callee(len(s))
	}
}

// calls.go T_calls_in_pseudo_loop 48 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// callsite: calls.go:50:9|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// callsite: calls.go:54:9|1 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_calls_in_pseudo_loop(x int, q []string) {
	for i := 0; i < x; i++ {
		callee(i)
		return
	}
	for _, s := range q {
		callee(len(s))
		break
	}
}

// calls.go T_calls_on_panic_paths 67 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// callsite: calls.go:69:9|0 flagstr "CallSiteOnPanicPath" flagval 2 score 42 mask 1 maskstr "panicPathAdj"
// callsite: calls.go:73:9|1 flagstr "CallSiteOnPanicPath" flagval 2 score 42 mask 1 maskstr "panicPathAdj"
// callsite: calls.go:77:12|2 flagstr "CallSiteOnPanicPath" flagval 2 score 102 mask 1 maskstr "panicPathAdj"
// <endcallsites>
// <endfuncpreamble>
func T_calls_on_panic_paths(x int, q []string) {
	if x+G == 101 {
		callee(x)
		panic("ouch")
	}
	if x < G-101 {
		callee(x)
		if len(q) == 0 {
			G++
		}
		callsexit(x)
	}
}

// calls.go T_calls_not_on_panic_paths 93 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch|ParamMayFeedIfOrSwitch
//   1 ParamNoInfo
// <endpropsdump>
// {"Flags":0,"ParamFlags":[96,0],"ResultFlags":null}
// callsite: calls.go:103:9|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// callsite: calls.go:112:9|1 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// callsite: calls.go:115:9|2 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// callsite: calls.go:119:12|3 flagstr "CallSiteOnPanicPath" flagval 2 score 102 mask 1 maskstr "panicPathAdj"
// <endcallsites>
// <endfuncpreamble>
func T_calls_not_on_panic_paths(x int, q []string) {
	if x != G {
		panic("ouch")
		/* Notes: */
		/* - we only look for post-dominating panic/exit, so */
		/*   this site will on fact not have a panicpath flag */
		/* - vet will complain about this site as unreachable */
		callee(x)
	}
	if x != G {
		callee(x)
		if x < 100 {
			panic("ouch")
		}
	}
	if x+G == 101 {
		if x < 100 {
			panic("ouch")
		}
		callee(x)
	}
	if x < -101 {
		callee(x)
		if len(q) == 0 {
			return
		}
		callsexit(x)
	}
}

// calls.go init.0 129 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// callsite: calls.go:130:16|0 flagstr "CallSiteInInitFunc" flagval 4 score 22 mask 2 maskstr "initFuncAdj"
// <endcallsites>
// <endfuncpreamble>
func init() {
	println(callee(5))
}

// calls.go T_pass_inlinable_func_to_param_feeding_indirect_call 140 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: calls.go:141:19|0 flagstr "" flagval 0 score -24 mask 512 maskstr "passInlinableFuncToIndCallAdj"
// callsite: calls.go:141:19|calls.go:231:10|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_pass_inlinable_func_to_param_feeding_indirect_call(x int) int {
	return callsParam(x, callee)
}

// calls.go T_pass_noninlinable_func_to_param_feeding_indirect_call 150 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: calls.go:153:19|0 flagstr "" flagval 0 score -4 mask 128 maskstr "passFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_pass_noninlinable_func_to_param_feeding_indirect_call(x int) int {
	// if we inline callsParam we can convert the indirect call
	// to a direct call, but we can't inline it.
	return callsParam(x, calleeNoInline)
}

// calls.go T_pass_inlinable_func_to_param_feeding_nested_indirect_call 165 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":[0]}
// callsite: calls.go:166:25|0 flagstr "" flagval 0 score -13 mask 1024 maskstr "passInlinableFuncToNestedIndCallAdj"
// callsite: calls.go:166:25|calls.go:236:11|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_pass_inlinable_func_to_param_feeding_nested_indirect_call(x int) int {
	return callsParamNested(x, callee)
}

// calls.go T_pass_noninlinable_func_to_param_feeding_nested_indirect_call 177 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":[0]}
// callsite: calls.go:178:25|0 flagstr "" flagval 0 score 7 mask 256 maskstr "passFuncToNestedIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_pass_noninlinable_func_to_param_feeding_nested_indirect_call(x int) int {
	return callsParamNested(x, calleeNoInline)
}

// calls.go T_call_scoring_in_noninlinable_func 194 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":[0]}
// callsite: calls.go:208:14|0 flagstr "CallSiteOnPanicPath" flagval 2 score 42 mask 1 maskstr "panicPathAdj"
// callsite: calls.go:209:15|1 flagstr "CallSiteOnPanicPath" flagval 2 score 42 mask 1 maskstr "panicPathAdj"
// callsite: calls.go:211:19|2 flagstr "" flagval 0 score -24 mask 512 maskstr "passInlinableFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
// calls.go T_call_scoring_in_noninlinable_func.func1 211 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// <endcallsites>
// <endfuncpreamble>
func T_call_scoring_in_noninlinable_func(x int, sl []int) int {
	if x == 101 {
		// Drive up the cost of inlining this funcfunc over the
		// regular threshold.
		for i := 0; i < 10; i++ {
			for j := 0; j < i; j++ {
				sl = append(sl, append(sl, append(sl, append(sl, x)...)...)...)
				sl = append(sl, sl[0], sl[1], sl[2])
				x += calleeNoInline(x)
			}
		}
	}
	if x < 100 {
		// make sure this callsite is scored properly
		G += callee(101)
		panic(callee(x))
	}
	return callsParam(x, func(y int) int { return y + x })
}

var G int

func callee(x int) int {
	return x
}

func calleeNoInline(x int) int {
	defer func() { G++ }()
	return x
}

func callsexit(x int) {
	println(x)
	os.Exit(x)
}

func callsParam(x int, f func(int) int) int {
	return f(x)
}

func callsParamNested(x int, f func(int) int) int {
	if x < 0 {
		return f(x)
	}
	return 0
}
