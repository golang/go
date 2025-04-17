// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>
package params

// acrosscall.go T_feeds_indirect_call_via_call_toplevel 19 0 1
// ParamFlags
//   0 ParamFeedsIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[8],"ResultFlags":null}
// callsite: acrosscall.go:20:12|0 flagstr "" flagval 0 score 20 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_indirect_call_via_call_toplevel(f func(int)) {
	callsparam(f)
}

// acrosscall.go T_feeds_indirect_call_via_call_conditional 31 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[16],"ResultFlags":null}
// callsite: acrosscall.go:33:13|0 flagstr "" flagval 0 score 20 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_indirect_call_via_call_conditional(f func(int)) {
	if G != 101 {
		callsparam(f)
	}
}

// acrosscall.go T_feeds_conditional_indirect_call_via_call_toplevel 45 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[16],"ResultFlags":null}
// callsite: acrosscall.go:46:23|0 flagstr "" flagval 0 score 24 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_conditional_indirect_call_via_call_toplevel(f func(int)) {
	callsparamconditional(f)
}

// acrosscall.go T_feeds_if_via_call 57 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":null}
// callsite: acrosscall.go:58:9|0 flagstr "" flagval 0 score 8 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_via_call(x int) {
	feedsif(x)
}

// acrosscall.go T_feeds_if_via_call_conditional 69 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64],"ResultFlags":null}
// callsite: acrosscall.go:71:10|0 flagstr "" flagval 0 score 8 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_if_via_call_conditional(x int) {
	if G != 101 {
		feedsif(x)
	}
}

// acrosscall.go T_feeds_conditional_if_via_call 83 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64],"ResultFlags":null}
// callsite: acrosscall.go:84:20|0 flagstr "" flagval 0 score 12 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_feeds_conditional_if_via_call(x int) {
	feedsifconditional(x)
}

// acrosscall.go T_multifeeds1 97 0 1
// ParamFlags
//   0 ParamFeedsIndirectCall|ParamMayFeedIndirectCall
//   1 ParamNoInfo
// <endpropsdump>
// {"Flags":0,"ParamFlags":[24,0],"ResultFlags":null}
// callsite: acrosscall.go:98:12|0 flagstr "" flagval 0 score 20 mask 0 maskstr ""
// callsite: acrosscall.go:99:23|1 flagstr "" flagval 0 score 24 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_multifeeds1(f1, f2 func(int)) {
	callsparam(f1)
	callsparamconditional(f1)
}

// acrosscall.go T_acrosscall_returnsconstant 110 0 1
// ResultFlags
//   0 ResultAlwaysSameConstant
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":[8]}
// callsite: acrosscall.go:111:24|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_acrosscall_returnsconstant() int {
	return returnsconstant()
}

// acrosscall.go T_acrosscall_returnsmem 122 0 1
// ResultFlags
//   0 ResultIsAllocatedMem
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":[2]}
// callsite: acrosscall.go:123:19|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_acrosscall_returnsmem() *int {
	return returnsmem()
}

// acrosscall.go T_acrosscall_returnscci 134 0 1
// ResultFlags
//   0 ResultIsConcreteTypeConvertedToInterface
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":[4]}
// callsite: acrosscall.go:135:19|0 flagstr "" flagval 0 score 7 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_acrosscall_returnscci() I {
	return returnscci()
}

// acrosscall.go T_acrosscall_multiret 144 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: acrosscall.go:146:25|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_acrosscall_multiret(q int) int {
	if q != G {
		return returnsconstant()
	}
	return 0
}

// acrosscall.go T_acrosscall_multiret2 158 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: acrosscall.go:160:25|0 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// callsite: acrosscall.go:162:25|1 flagstr "" flagval 0 score 2 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_acrosscall_multiret2(q int) int {
	if q == G {
		return returnsconstant()
	} else {
		return returnsconstant()
	}
}

func callsparam(f func(int)) {
	f(2)
}

func callsparamconditional(f func(int)) {
	if G != 101 {
		f(2)
	}
}

func feedsif(x int) int {
	if x != 101 {
		return 42
	}
	return 43
}

func feedsifconditional(x int) int {
	if G != 101 {
		if x != 101 {
			return 42
		}
	}
	return 43
}

func returnsconstant() int {
	return 42
}

func returnsmem() *int {
	return new(int)
}

func returnscci() I {
	var q Q
	return q
}

type I interface {
	Foo()
}

type Q int

func (q Q) Foo() {
}

var G int
