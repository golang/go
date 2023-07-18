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
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[]}
// callsite: calls.go:21:15|0 flagstr "CallSiteOnPanicPath" flagval 2
// <endcallsites>
// <endfuncpreamble>
func T_call_in_panic_arg(x int) {
	if x < G {
		panic(callee(x))
	}
}

// calls.go T_calls_in_loops 32 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":[]}
// callsite: calls.go:34:9|0 flagstr "CallSiteInLoop" flagval 1
// callsite: calls.go:37:9|1 flagstr "CallSiteInLoop" flagval 1
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
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":[]}
// callsite: calls.go:50:9|0 flagstr "" flagval 0
// callsite: calls.go:54:9|1 flagstr "" flagval 0
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
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":[]}
// callsite: calls.go:69:9|0 flagstr "" flagval 0
// callsite: calls.go:73:9|1 flagstr "" flagval 0
// callsite: calls.go:77:12|2 flagstr "CallSiteOnPanicPath" flagval 2
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
// {"Flags":0,"ParamFlags":[96,0],"ResultFlags":[]}
// callsite: calls.go:103:9|0 flagstr "" flagval 0
// callsite: calls.go:112:9|1 flagstr "" flagval 0
// callsite: calls.go:115:9|2 flagstr "" flagval 0
// callsite: calls.go:119:12|3 flagstr "" flagval 0
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
// {"Flags":0,"ParamFlags":[],"ResultFlags":[]}
// callsite: calls.go:130:16|0 flagstr "CallSiteInInitFunc" flagval 4
// <endcallsites>
// <endfuncpreamble>
func init() {
	println(callee(5))
}

var G int

func callee(x int) int {
	return x
}

func callsexit(x int) {
	println(x)
	os.Exit(x)
}
