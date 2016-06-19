// +build amd64
// errorcheck -0 -l -live -wb=0

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// liveness tests with inlining disabled.
// see also live2.go.

package main

func printnl()

//go:noescape
func printpointer(**int)

//go:noescape
func printintpointer(*int)

//go:noescape
func printstringpointer(*string)

//go:noescape
func printstring(string)

//go:noescape
func printbytepointer(*byte)

func printint(int)

func f1() {
	var x *int
	printpointer(&x) // ERROR "live at call to printpointer: x$"
	printpointer(&x) // ERROR "live at call to printpointer: x$"
}

func f2(b bool) {
	if b {
		printint(0) // nothing live here
		return
	}
	var x *int
	printpointer(&x) // ERROR "live at call to printpointer: x$"
	printpointer(&x) // ERROR "live at call to printpointer: x$"
}

func f3(b1, b2 bool) {
	// Because x and y are ambiguously live, they appear
	// live throughout the function, to avoid being poisoned
	// in GODEBUG=gcdead=1 mode.

	printint(0) // ERROR "live at call to printint: x y$"
	if b1 == false {
		printint(0) // ERROR "live at call to printint: x y$"
		return
	}

	if b2 {
		var x *int
		printpointer(&x) // ERROR "live at call to printpointer: x y$"
		printpointer(&x) // ERROR "live at call to printpointer: x y$"
	} else {
		var y *int
		printpointer(&y) // ERROR "live at call to printpointer: x y$"
		printpointer(&y) // ERROR "live at call to printpointer: x y$"
	}
	printint(0) // ERROR "f3: x \(type \*int\) is ambiguously live$" "f3: y \(type \*int\) is ambiguously live$" "live at call to printint: x y$"
}

// The old algorithm treated x as live on all code that
// could flow to a return statement, so it included the
// function entry and code above the declaration of x
// but would not include an indirect use of x in an infinite loop.
// Check that these cases are handled correctly.

func f4(b1, b2 bool) { // x not live here
	if b2 {
		printint(0) // x not live here
		return
	}
	var z **int
	x := new(int)
	*x = 42
	z = &x
	printint(**z) // ERROR "live at call to printint: x$"
	if b2 {
		printint(1) // x not live here
		return
	}
	for {
		printint(**z) // ERROR "live at call to printint: x$"
	}
}

func f5(b1 bool) {
	var z **int
	if b1 {
		x := new(int)
		*x = 42
		z = &x
	} else {
		y := new(int)
		*y = 54
		z = &y
	}
	printint(**z) // ERROR "f5: x \(type \*int\) is ambiguously live$" "f5: y \(type \*int\) is ambiguously live$" "live at call to printint: x y$"
}

// confusion about the _ result used to cause spurious "live at entry to f6: _".

func f6() (_, y string) {
	y = "hello"
	return
}

// confusion about addressed results used to cause "live at entry to f7: x".

func f7() (x string) {
	_ = &x
	x = "hello"
	return
}

// ignoring block returns used to cause "live at entry to f8: x, y".

func f8() (x, y string) {
	return g8()
}

func g8() (string, string)

// ignoring block assignments used to cause "live at entry to f9: x"
// issue 7205

var i9 interface{}

func f9() bool {
	g8()
	x := i9
	return x != interface{}(99.0i) // ERROR "live at call to convT2E: x.data x.type$"
}

// liveness formerly confused by UNDEF followed by RET,
// leading to "live at entry to f10: ~r1" (unnamed result).

func f10() string {
	panic(1)
}

// liveness formerly confused by select, thinking runtime.selectgo
// can return to next instruction; it always jumps elsewhere.
// note that you have to use at least two cases in the select
// to get a true select; smaller selects compile to optimized helper functions.

var c chan *int
var b bool

// this used to have a spurious "live at entry to f11a: ~r0"
func f11a() *int {
	select { // ERROR "live at call to newselect: autotmp_[0-9]+$" "live at call to selectgo: autotmp_[0-9]+$"
	case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+$"
		return nil
	case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+$"
		return nil
	}
}

func f11b() *int {
	p := new(int)
	if b {
		// At this point p is dead: the code here cannot
		// get to the bottom of the function.
		// This used to have a spurious "live at call to printint: p".
		printint(1) // nothing live here!
		select {    // ERROR "live at call to newselect: autotmp_[0-9]+$" "live at call to selectgo: autotmp_[0-9]+$"
		case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+$"
			return nil
		case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+$"
			return nil
		}
	}
	println(*p)
	return nil
}

var sink *int

func f11c() *int {
	p := new(int)
	sink = p // prevent stack allocation, otherwise p is rematerializeable
	if b {
		// Unlike previous, the cases in this select fall through,
		// so we can get to the println, so p is not dead.
		printint(1) // ERROR "live at call to printint: p$"
		select {    // ERROR "live at call to newselect: autotmp_[0-9]+ p$" "live at call to selectgo: autotmp_[0-9]+ p$"
		case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+ p$"
		case <-c: // ERROR "live at call to selectrecv: autotmp_[0-9]+ p$"
		}
	}
	println(*p)
	return nil
}

// similarly, select{} does not fall through.
// this used to have a spurious "live at entry to f12: ~r0".

func f12() *int {
	if b {
		select {}
	} else {
		return nil
	}
}

// incorrectly placed VARDEF annotations can cause missing liveness annotations.
// this used to be missing the fact that s is live during the call to g13 (because it is
// needed for the call to h13).

func f13() {
	s := g14()
	s = h13(s, g13(s)) // ERROR "live at call to g13: s.ptr$"
}

func g13(string) string
func h13(string, string) string

// more incorrectly placed VARDEF.

func f14() {
	x := g14()
	printstringpointer(&x) // ERROR "live at call to printstringpointer: x$"
}

func g14() string

func f15() {
	var x string
	_ = &x
	x = g15()      // ERROR "live at call to g15: x$"
	printstring(x) // ERROR "live at call to printstring: x$"
}

func g15() string

// Checking that various temporaries do not persist or cause
// ambiguously live values that must be zeroed.
// The exact temporary names are inconsequential but we are
// trying to check that there is only one at any given site,
// and also that none show up in "ambiguously live" messages.

var m map[string]int

func f16() {
	if b {
		delete(m, "hi") // ERROR "live at call to mapdelete: autotmp_[0-9]+$"
	}
	delete(m, "hi") // ERROR "live at call to mapdelete: autotmp_[0-9]+$"
	delete(m, "hi") // ERROR "live at call to mapdelete: autotmp_[0-9]+$"
}

var m2s map[string]*byte
var m2 map[[2]string]*byte
var x2 [2]string
var bp *byte

func f17a() {
	// value temporary only
	if b {
		m2[x2] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
	}
	m2[x2] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
	m2[x2] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
}

func f17b() {
	// key temporary only
	if b {
		m2s["x"] = bp // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
	}
	m2s["x"] = bp // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
	m2s["x"] = bp // ERROR "live at call to mapassign1: autotmp_[0-9]+$"
}

func f17c() {
	// key and value temporaries
	if b {
		m2s["x"] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
	}
	m2s["x"] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
	m2s["x"] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
}

func g18() [2]string

func f18() {
	// key temporary for mapaccess.
	// temporary introduced by orderexpr.
	var z *byte
	if b {
		z = m2[g18()] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	}
	z = m2[g18()] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	z = m2[g18()] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	printbytepointer(z)
}

var ch chan *byte

func f19() {
	// dest temporary for channel receive.
	var z *byte

	if b {
		z = <-ch // ERROR "live at call to chanrecv1: autotmp_[0-9]+$"
	}
	z = <-ch // ERROR "live at call to chanrecv1: autotmp_[0-9]+$"
	z = <-ch // ERROR "live at call to chanrecv1: autotmp_[0-9]+$"
	printbytepointer(z)
}

func f20() {
	// src temporary for channel send
	if b {
		ch <- nil // ERROR "live at call to chansend1: autotmp_[0-9]+$"
	}
	ch <- nil // ERROR "live at call to chansend1: autotmp_[0-9]+$"
	ch <- nil // ERROR "live at call to chansend1: autotmp_[0-9]+$"
}

func f21() {
	// key temporary for mapaccess using array literal key.
	var z *byte
	if b {
		z = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	}
	z = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	z = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
	printbytepointer(z)
}

func f23() {
	// key temporary for two-result map access using array literal key.
	var z *byte
	var ok bool
	if b {
		z, ok = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess2: autotmp_[0-9]+$"
	}
	z, ok = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess2: autotmp_[0-9]+$"
	z, ok = m2[[2]string{"x", "y"}] // ERROR "live at call to mapaccess2: autotmp_[0-9]+$"
	printbytepointer(z)
	print(ok)
}

func f24() {
	// key temporary for map access using array literal key.
	// value temporary too.
	if b {
		m2[[2]string{"x", "y"}] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
	}
	m2[[2]string{"x", "y"}] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
	m2[[2]string{"x", "y"}] = nil // ERROR "live at call to mapassign1: autotmp_[0-9]+ autotmp_[0-9]+$"
}

// defer should not cause spurious ambiguously live variables

func f25(b bool) {
	defer g25()
	if b {
		return
	}
	var x string
	_ = &x
	x = g15()      // ERROR "live at call to g15: x$"
	printstring(x) // ERROR "live at call to printstring: x$"
} // ERROR "live at call to deferreturn: x$"

func g25()

// non-escaping ... slices passed to function call should die on return,
// so that the temporaries do not stack and do not cause ambiguously
// live variables.

func f26(b bool) {
	if b {
		print26((*int)(nil), (*int)(nil), (*int)(nil)) // ERROR "live at call to print26: autotmp_[0-9]+$"
	}
	print26((*int)(nil), (*int)(nil), (*int)(nil)) // ERROR "live at call to print26: autotmp_[0-9]+$"
	print26((*int)(nil), (*int)(nil), (*int)(nil)) // ERROR "live at call to print26: autotmp_[0-9]+$"
	printnl()
}

//go:noescape
func print26(...interface{})

// non-escaping closures passed to function call should die on return

func f27(b bool) {
	x := 0
	if b {
		call27(func() { x++ }) // ERROR "live at call to call27: autotmp_[0-9]+$"
	}
	call27(func() { x++ }) // ERROR "live at call to call27: autotmp_[0-9]+$"
	call27(func() { x++ }) // ERROR "live at call to call27: autotmp_[0-9]+$"
	printnl()
}

// but defer does escape to later execution in the function

func f27defer(b bool) {
	x := 0
	if b {
		defer call27(func() { x++ }) // ERROR "live at call to deferproc: autotmp_[0-9]+$" "live at call to deferreturn: autotmp_[0-9]+$"
	}
	defer call27(func() { x++ }) // ERROR "f27defer: autotmp_[0-9]+ \(type struct { F uintptr; x \*int }\) is ambiguously live$" "live at call to deferproc: autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to deferreturn: autotmp_[0-9]+ autotmp_[0-9]+$"
	printnl()                    // ERROR "live at call to printnl: autotmp_[0-9]+ autotmp_[0-9]+$"
} // ERROR "live at call to deferreturn: autotmp_[0-9]+ autotmp_[0-9]+$"

// and newproc (go) escapes to the heap

func f27go(b bool) {
	x := 0
	if b {
		go call27(func() { x++ }) // ERROR "live at call to newobject: &x$" "live at call to newproc: &x$"
	}
	go call27(func() { x++ }) // ERROR "live at call to newobject: &x$"
	printnl()
}

//go:noescape
func call27(func())

// concatstring slice should die on return

var s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 string

func f28(b bool) {
	if b {
		printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10) // ERROR "live at call to concatstrings: autotmp_[0-9]+$" "live at call to printstring: autotmp_[0-9]+$"
	}
	printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10) // ERROR "live at call to concatstrings: autotmp_[0-9]+$" "live at call to printstring: autotmp_[0-9]+$"
	printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10) // ERROR "live at call to concatstrings: autotmp_[0-9]+$" "live at call to printstring: autotmp_[0-9]+$"
}

// map iterator should die on end of range loop

func f29(b bool) {
	if b {
		for k := range m { // ERROR "live at call to mapiterinit: autotmp_[0-9]+$" "live at call to mapiternext: autotmp_[0-9]+$"
			printstring(k) // ERROR "live at call to printstring: autotmp_[0-9]+$"
		}
	}
	for k := range m { // ERROR "live at call to mapiterinit: autotmp_[0-9]+$" "live at call to mapiternext: autotmp_[0-9]+$"
		printstring(k) // ERROR "live at call to printstring: autotmp_[0-9]+$"
	}
	for k := range m { // ERROR "live at call to mapiterinit: autotmp_[0-9]+$" "live at call to mapiternext: autotmp_[0-9]+$"
		printstring(k) // ERROR "live at call to printstring: autotmp_[0-9]+$"
	}
}

// copy of array of pointers should die at end of range loop

var ptrarr [10]*int

func f30(b bool) {
	// two live temps during print(p):
	// the copy of ptrarr and the internal iterator pointer.
	if b {
		for _, p := range ptrarr {
			printintpointer(p) // ERROR "live at call to printintpointer: autotmp_[0-9]+ autotmp_[0-9]+$"
		}
	}
	for _, p := range ptrarr {
		printintpointer(p) // ERROR "live at call to printintpointer: autotmp_[0-9]+ autotmp_[0-9]+$"
	}
	for _, p := range ptrarr {
		printintpointer(p) // ERROR "live at call to printintpointer: autotmp_[0-9]+ autotmp_[0-9]+$"
	}
}

// conversion to interface should not leave temporary behind

func f31(b1, b2, b3 bool) {
	if b1 {
		g31("a") // ERROR "live at call to convT2E: autotmp_[0-9]+$" "live at call to g31: autotmp_[0-9]+$"
	}
	if b2 {
		h31("b") // ERROR "live at call to convT2E: autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to h31: autotmp_[0-9]+$" "live at call to newobject: autotmp_[0-9]+$"
	}
	if b3 {
		panic("asdf") // ERROR "live at call to convT2E: autotmp_[0-9]+$" "live at call to gopanic: autotmp_[0-9]+$"
	}
	print(b3)
}

func g31(interface{})
func h31(...interface{})

// non-escaping partial functions passed to function call should die on return

type T32 int

func (t *T32) Inc() { // ERROR "live at entry to \(\*T32\).Inc: t$"
	*t++
}

var t32 T32

func f32(b bool) {
	if b {
		call32(t32.Inc) // ERROR "live at call to call32: autotmp_[0-9]+$"
	}
	call32(t32.Inc) // ERROR "live at call to call32: autotmp_[0-9]+$"
	call32(t32.Inc) // ERROR "live at call to call32: autotmp_[0-9]+$"
}

//go:noescape
func call32(func())

// temporaries introduced during if conditions and && || expressions
// should die once the condition has been acted upon.

var m33 map[interface{}]int

func f33() {
	if m33[nil] == 0 { // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
		printnl()
		return
	} else {
		printnl()
	}
	printnl()
}

func f34() {
	if m33[nil] == 0 { // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
		printnl()
		return
	}
	printnl()
}

func f35() {
	if m33[nil] == 0 && m33[nil] == 0 { // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
		printnl()
		return
	}
	printnl()
}

func f36() {
	if m33[nil] == 0 || m33[nil] == 0 { // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
		printnl()
		return
	}
	printnl()
}

func f37() {
	if (m33[nil] == 0 || m33[nil] == 0) && m33[nil] == 0 { // ERROR "live at call to mapaccess1: autotmp_[0-9]+$"
		printnl()
		return
	}
	printnl()
}

// select temps should disappear in the case bodies

var c38 chan string

func fc38() chan string
func fi38(int) *string
func fb38() *bool

func f38(b bool) {
	// we don't care what temps are printed on the lines with output.
	// we care that the println lines have no live variables
	// and therefore no output.
	if b {
		select { // ERROR "live at call to newselect: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to selectgo: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$"
		case <-fc38(): // ERROR "live at call to selectrecv: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$"
			printnl()
		case fc38() <- *fi38(1): // ERROR "live at call to fc38: autotmp_[0-9]+$" "live at call to fi38: autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to selectsend: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$"
			printnl()
		case *fi38(2) = <-fc38(): // ERROR "live at call to fc38: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to fi38: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to selectrecv: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$"
			printnl()
		case *fi38(3), *fb38() = <-fc38(): // ERROR "live at call to fb38: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to fc38: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to fi38: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$" "live at call to selectrecv2: autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+ autotmp_[0-9]+$"
			printnl()
		}
		printnl()
	}
	printnl()
}

// issue 8097: mishandling of x = x during return.

func f39() (x []int) {
	x = []int{1}
	printnl() // ERROR "live at call to printnl: autotmp_[0-9]+$"
	return x
}

func f39a() (x []int) {
	x = []int{1}
	printnl() // ERROR "live at call to printnl: autotmp_[0-9]+$"
	return
}

func f39b() (x [10]*int) {
	x = [10]*int{}
	x[0] = new(int) // ERROR "live at call to newobject: x$"
	printnl()       // ERROR "live at call to printnl: x$"
	return x
}

func f39c() (x [10]*int) {
	x = [10]*int{}
	x[0] = new(int) // ERROR "live at call to newobject: x$"
	printnl()       // ERROR "live at call to printnl: x$"
	return
}

// issue 8142: lost 'addrtaken' bit on inlined variables.
// no inlining in this test, so just checking that non-inlined works.

type T40 struct {
	m map[int]int
}

func newT40() *T40 {
	ret := T40{}
	ret.m = make(map[int]int) // ERROR "live at call to makemap: &ret$"
	return &ret
}

func bad40() {
	t := newT40()
	_ = t
	printnl()
}

func good40() {
	ret := T40{}
	ret.m = make(map[int]int) // ERROR "live at call to makemap: autotmp_[0-9]+ ret$"
	t := &ret
	printnl() // ERROR "live at call to printnl: autotmp_[0-9]+ ret$"
	_ = t
}
