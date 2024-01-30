// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>

package returns2

// returns2.go T_return_feeds_iface_call 18 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// callsite: returns2.go:19:13|0 flagstr "" flagval 0 score 1 mask 16384 maskstr "returnFeedsConcreteToInterfaceCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_return_feeds_iface_call() {
	b := newBar(10)
	b.Plark()
}

// returns2.go T_multi_return_feeds_iface_call 29 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// callsite: returns2.go:30:20|0 flagstr "" flagval 0 score 3 mask 16384 maskstr "returnFeedsConcreteToInterfaceCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_multi_return_feeds_iface_call() {
	_, b, _ := newBar2(10)
	b.Plark()
}

// returns2.go T_returned_inlinable_func_feeds_indirect_call 41 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// callsite: returns2.go:42:18|0 flagstr "" flagval 0 score -51 mask 8200 maskstr "passConstToIfAdj|returnFeedsInlinableFuncToIndCallAdj"
// callsite: returns2.go:44:20|1 flagstr "" flagval 0 score -23 mask 8192 maskstr "returnFeedsInlinableFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_returned_inlinable_func_feeds_indirect_call(q int) {
	f := returnsFunc(10)
	f(q)
	f2 := returnsFunc2()
	f2(q)
}

// returns2.go T_returned_noninlineable_func_feeds_indirect_call 54 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// callsite: returns2.go:55:30|0 flagstr "" flagval 0 score -23 mask 4096 maskstr "returnFeedsFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_returned_noninlineable_func_feeds_indirect_call(q int) {
	f := returnsNonInlinableFunc()
	f(q)
}

// returns2.go T_multi_return_feeds_indirect_call 65 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":null}
// callsite: returns2.go:66:29|0 flagstr "" flagval 0 score -21 mask 8192 maskstr "returnFeedsInlinableFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_multi_return_feeds_indirect_call(q int) {
	_, f, _ := multiReturnsFunc()
	f(q)
}

// returns2.go T_return_feeds_ifswitch 76 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: returns2.go:77:14|0 flagstr "" flagval 0 score 10 mask 2048 maskstr "returnFeedsConstToIfAdj"
// <endcallsites>
// <endfuncpreamble>
func T_return_feeds_ifswitch(q int) int {
	x := meaning(q)
	if x < 42 {
		switch x {
		case 42:
			return 1
		}
	}
	return 0
}

// returns2.go T_multi_return_feeds_ifswitch 93 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: returns2.go:94:21|0 flagstr "" flagval 0 score 9 mask 2048 maskstr "returnFeedsConstToIfAdj"
// <endcallsites>
// <endfuncpreamble>
func T_multi_return_feeds_ifswitch(q int) int {
	x, y, z := meanings(q)
	if x < y {
		switch x {
		case 42:
			return z
		}
	}
	return 0
}

// returns2.go T_two_calls_feed_ifswitch 111 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0],"ResultFlags":[0]}
// callsite: returns2.go:115:14|0 flagstr "" flagval 0 score 25 mask 0 maskstr ""
// callsite: returns2.go:116:14|1 flagstr "" flagval 0 score 25 mask 0 maskstr ""
// <endcallsites>
// <endfuncpreamble>
func T_two_calls_feed_ifswitch(q int) int {
	// This case we don't handle; for the heuristic to kick in,
	// all names in a given if/switch cond have to come from the
	// same callsite
	x := meaning(q)
	y := meaning(-q)
	if x < y {
		switch x + y {
		case 42:
			return 1
		}
	}
	return 0
}

// returns2.go T_chained_indirect_call 132 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// callsite: returns2.go:135:18|0 flagstr "" flagval 0 score -31 mask 8192 maskstr "returnFeedsInlinableFuncToIndCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_chained_indirect_call(x, y int) {
	// Here 'returnsFunc' returns an inlinable func that feeds
	// directly into a call (no named intermediate).
	G += returnsFunc(x - y)(x + y)
}

// returns2.go T_chained_conc_iface_call 144 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":[0,0],"ResultFlags":null}
// callsite: returns2.go:148:8|0 flagstr "" flagval 0 score 1 mask 16384 maskstr "returnFeedsConcreteToInterfaceCallAdj"
// <endcallsites>
// <endfuncpreamble>
func T_chained_conc_iface_call(x, y int) {
	// Similar to the case above, return from call returning concrete type
	// feeds directly into interface call. Note that only the first
	// iface call is interesting here.
	newBar(10).Plark().Plark()
}

func returnsFunc(x int) func(int) int {
	if x < 0 {
		G++
	}
	return adder
}

func returnsFunc2() func(int) int {
	return func(x int) int {
		return adder(x)
	}
}

func returnsNonInlinableFunc() func(int) int {
	return adderNoInline
}

func multiReturnsFunc() (int, func(int) int, int) {
	return 42, func(x int) int { G++; return 1 }, -42
}

func adder(x int) int {
	G += 1
	return G
}

func adderNoInline(x int) int {
	defer func() { G += x }()
	G += 1
	return G
}

func meaning(q int) int {
	r := 0
	for i := 0; i < 42; i++ {
		r += q
	}
	G += r
	return 42
}

func meanings(q int) (int, int, int) {
	r := 0
	for i := 0; i < 42; i++ {
		r += q
	}
	return 42, 43, r
}

type Bar struct {
	x int
	y string
}

func (b *Bar) Plark() Itf {
	return b
}

type Itf interface {
	Plark() Itf
}

func newBar(x int) Itf {
	s := 0
	for i := 0; i < x; i++ {
		s += i
	}
	return &Bar{
		x: s,
	}
}

func newBar2(x int) (int, Itf, bool) {
	s := 0
	for i := 0; i < x; i++ {
		s += i
	}
	return 0, &Bar{x: s}, false
}

var G int
