// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>
package params

// acrosscall.go T_feeds_indirect_call_via_call_toplevel 17 0 1
// ParamFlags
//   0 ParamFeedsIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[8],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_indirect_call_via_call_toplevel(f func(int)) {
	callsparam(f)
}

// acrosscall.go T_feeds_indirect_call_via_call_conditional 27 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[16],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_indirect_call_via_call_conditional(f func(int)) {
	if G != 101 {
		callsparam(f)
	}
}

// acrosscall.go T_feeds_conditional_indirect_call_via_call_toplevel 39 0 1
// ParamFlags
//   0 ParamMayFeedIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[16],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_conditional_indirect_call_via_call_toplevel(f func(int)) {
	callsparamconditional(f)
}

// acrosscall.go T_feeds_if_via_call 49 0 1
// ParamFlags
//   0 ParamFeedsIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[32],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_if_via_call(x int) {
	feedsif(x)
}

// acrosscall.go T_feeds_if_via_call_conditional 59 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_if_via_call_conditional(x int) {
	if G != 101 {
		feedsif(x)
	}
}

// acrosscall.go T_feeds_conditional_if_via_call 71 0 1
// ParamFlags
//   0 ParamMayFeedIfOrSwitch
// <endpropsdump>
// {"Flags":0,"ParamFlags":[64],"ResultFlags":[]}
// <endfuncpreamble>
func T_feeds_conditional_if_via_call(x int) {
	feedsifconditional(x)
}

// acrosscall.go T_multifeeds 82 0 1
// ParamFlags
//   0 ParamFeedsIndirectCall|ParamMayFeedIndirectCall
//   1 ParamFeedsIndirectCall
// <endpropsdump>
// {"Flags":0,"ParamFlags":[24,8],"ResultFlags":[]}
// <endfuncpreamble>
func T_multifeeds(f1, f2 func(int)) {
	callsparam(f1)
	callsparamconditional(f1)
	callsparam(f2)
}

// acrosscall.go T_acrosscall_returnsconstant 94 0 1
// ResultFlags
//   0 ResultAlwaysSameConstant
// <endpropsdump>
// {"Flags":0,"ParamFlags":[],"ResultFlags":[8]}
// <endfuncpreamble>
func T_acrosscall_returnsconstant() int {
	return returnsconstant()
}

// acrosscall.go T_acrosscall_returnsmem 104 0 1
// ResultFlags
//   0 ResultIsAllocatedMem
// <endpropsdump>
// {"Flags":0,"ParamFlags":[],"ResultFlags":[2]}
// <endfuncpreamble>
func T_acrosscall_returnsmem() *int {
	return returnsmem()
}

// acrosscall.go T_acrosscall_returnscci 114 0 1
// ResultFlags
//   0 ResultIsConcreteTypeConvertedToInterface
// <endpropsdump>
// {"Flags":0,"ParamFlags":[],"ResultFlags":[4]}
// <endfuncpreamble>
func T_acrosscall_returnscci() I {
	return returnscci()
}

// acrosscall.go T_acrosscall_multiret 122 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// <endfuncpreamble>
func T_acrosscall_multiret(q int) int {
	if q != G {
		return returnsconstant()
	}
	return 0
}

// acrosscall.go T_acrosscall_multiret2 133 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
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
