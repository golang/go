// errorcheckwithauto -0 -l -live -wb=0 -d=ssa/insert_resched_checks/off

//go:build (amd64 && goexperiment.regabiargs) || (arm64 && goexperiment.regabiargs)

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// liveness tests with inlining disabled.
// see also live2.go.

package main

import "runtime"

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
	var x *int       // ERROR "stack object x \*int$"
	printpointer(&x) // ERROR "live at call to printpointer: x$"
	printpointer(&x)
}

func f2(b bool) {
	if b {
		printint(0) // nothing live here
		return
	}
	var x *int       // ERROR "stack object x \*int$"
	printpointer(&x) // ERROR "live at call to printpointer: x$"
	printpointer(&x)
}

func f3(b1, b2 bool) {
	// Here x and y are ambiguously live. In previous go versions they
	// were marked as live throughout the function to avoid being
	// poisoned in GODEBUG=gcdead=1 mode; this is now no longer the
	// case.

	printint(0)
	if b1 == false {
		printint(0)
		return
	}

	if b2 {
		var x *int       // ERROR "stack object x \*int$"
		printpointer(&x) // ERROR "live at call to printpointer: x$"
		printpointer(&x)
	} else {
		var y *int       // ERROR "stack object y \*int$"
		printpointer(&y) // ERROR "live at call to printpointer: y$"
		printpointer(&y)
	}
	printint(0) // nothing is live here
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
	x := new(int) // ERROR "stack object x \*int$"
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
		x := new(int) // ERROR "stack object x \*int$"
		*x = 42
		z = &x
	} else {
		y := new(int) // ERROR "stack object y \*int$"
		*y = 54
		z = &y
	}
	printint(**z) // nothing live here
}

// confusion about the _ result used to cause spurious "live at entry to f6: _".

func f6() (_, y string) {
	y = "hello"
	return
}

// confusion about addressed results used to cause "live at entry to f7: x".

func f7() (x string) { // ERROR "stack object x string"
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
	y := interface{}(g18()) // ERROR "live at call to convT: x.data$" "live at call to g18: x.data$" "stack object .autotmp_[0-9]+ \[2\]string$"
	i9 = y                  // make y escape so the line above has to call convT
	return x != y
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
	select { // ERROR "stack object .autotmp_[0-9]+ \[2\]runtime.scase$"
	case <-c:
		return nil
	case <-c:
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
		select {    // ERROR "stack object .autotmp_[0-9]+ \[2\]runtime.scase$"
		case <-c:
			return nil
		case <-c:
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
		select {    // ERROR "live at call to selectgo: p$" "stack object .autotmp_[0-9]+ \[2\]runtime.scase$"
		case <-c:
		case <-c:
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
	x := g14() // ERROR "stack object x string$"
	printstringpointer(&x)
}

func g14() string

// Checking that various temporaries do not persist or cause
// ambiguously live values that must be zeroed.
// The exact temporary names are inconsequential but we are
// trying to check that there is only one at any given site,
// and also that none show up in "ambiguously live" messages.

var m map[string]int
var mi map[interface{}]int

// str and iface are used to ensure that a temp is required for runtime calls below.
func str() string
func iface() interface{}

func f16() {
	if b {
		delete(mi, iface()) // ERROR "stack object .autotmp_[0-9]+ interface \{\}$"
	}
	delete(mi, iface())
	delete(mi, iface())
}

var m2s map[string]*byte
var m2 map[[2]string]*byte
var x2 [2]string
var bp *byte

func f17a(p *byte) { // ERROR "live at entry to f17a: p$"
	if b {
		m2[x2] = p // ERROR "live at call to mapassign: p$"
	}
	m2[x2] = p // ERROR "live at call to mapassign: p$"
	m2[x2] = p // ERROR "live at call to mapassign: p$"
}

func f17b(p *byte) { // ERROR "live at entry to f17b: p$"
	// key temporary
	if b {
		m2s[str()] = p // ERROR "live at call to mapassign_faststr: p$" "live at call to str: p$"

	}
	m2s[str()] = p // ERROR "live at call to mapassign_faststr: p$" "live at call to str: p$"
	m2s[str()] = p // ERROR "live at call to mapassign_faststr: p$" "live at call to str: p$"
}

func f17c() {
	// key and value temporaries
	if b {
		m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign_faststr: .autotmp_[0-9]+$"
	}
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign_faststr: .autotmp_[0-9]+$"
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign_faststr: .autotmp_[0-9]+$"
}

func f17d() *byte

func g18() [2]string

func f18() {
	// key temporary for mapaccess.
	// temporary introduced by orderexpr.
	var z *byte
	if b {
		z = m2[g18()] // ERROR "stack object .autotmp_[0-9]+ \[2\]string$"
	}
	z = m2[g18()]
	z = m2[g18()]
	printbytepointer(z)
}

var ch chan *byte

// byteptr is used to ensure that a temp is required for runtime calls below.
func byteptr() *byte

func f19() {
	// dest temporary for channel receive.
	var z *byte

	if b {
		z = <-ch // ERROR "stack object .autotmp_[0-9]+ \*byte$"
	}
	z = <-ch
	z = <-ch // ERROR "live at call to chanrecv1: .autotmp_[0-9]+$"
	printbytepointer(z)
}

func f20() {
	// src temporary for channel send
	if b {
		ch <- byteptr() // ERROR "stack object .autotmp_[0-9]+ \*byte$"
	}
	ch <- byteptr()
	ch <- byteptr()
}

func f21(x, y string) { // ERROR "live at entry to f21: x y"
	// key temporary for mapaccess using array literal key.
	var z *byte
	if b {
		z = m2[[2]string{x, y}] // ERROR "stack object .autotmp_[0-9]+ \[2\]string$"
	}
	z = m2[[2]string{"x", "y"}]
	z = m2[[2]string{"x", "y"}]
	printbytepointer(z)
}

func f21b() {
	// key temporary for mapaccess using array literal key.
	var z *byte
	if b {
		z = m2[[2]string{"x", "y"}]
	}
	z = m2[[2]string{"x", "y"}]
	z = m2[[2]string{"x", "y"}]
	printbytepointer(z)
}

func f23(x, y string) { // ERROR "live at entry to f23: x y"
	// key temporary for two-result map access using array literal key.
	var z *byte
	var ok bool
	if b {
		z, ok = m2[[2]string{x, y}] // ERROR "stack object .autotmp_[0-9]+ \[2\]string$"
	}
	z, ok = m2[[2]string{"x", "y"}]
	z, ok = m2[[2]string{"x", "y"}]
	printbytepointer(z)
	print(ok)
}

func f23b() {
	// key temporary for two-result map access using array literal key.
	var z *byte
	var ok bool
	if b {
		z, ok = m2[[2]string{"x", "y"}]
	}
	z, ok = m2[[2]string{"x", "y"}]
	z, ok = m2[[2]string{"x", "y"}]
	printbytepointer(z)
	print(ok)
}

func f24(x, y string) { // ERROR "live at entry to f24: x y"
	// key temporary for map access using array lit3ral key.
	// value temporary too.
	if b {
		m2[[2]string{x, y}] = nil // ERROR "stack object .autotmp_[0-9]+ \[2\]string$"
	}
	m2[[2]string{"x", "y"}] = nil
	m2[[2]string{"x", "y"}] = nil
}

func f24b() {
	// key temporary for map access using array literal key.
	// value temporary too.
	if b {
		m2[[2]string{"x", "y"}] = nil
	}
	m2[[2]string{"x", "y"}] = nil
	m2[[2]string{"x", "y"}] = nil
}

// Non-open-coded defers should not cause autotmps.  (Open-coded defers do create extra autotmps).
func f25(b bool) {
	for i := 0; i < 2; i++ {
		// Put in loop to make sure defer is not open-coded
		defer g25()
	}
	if b {
		return
	}
	var x string
	x = g14()
	printstring(x)
	return
}

func g25()

// non-escaping ... slices passed to function call should die on return,
// so that the temporaries do not stack and do not cause ambiguously
// live variables.

func f26(b bool) {
	if b {
		print26((*int)(nil), (*int)(nil), (*int)(nil)) // ERROR "stack object .autotmp_[0-9]+ \[3\]interface \{\}$"
	}
	print26((*int)(nil), (*int)(nil), (*int)(nil))
	print26((*int)(nil), (*int)(nil), (*int)(nil))
	printnl()
}

//go:noescape
func print26(...interface{})

// non-escaping closures passed to function call should die on return

func f27(b bool) {
	x := 0
	if b {
		call27(func() { x++ }) // ERROR "stack object .autotmp_[0-9]+ struct \{"
	}
	call27(func() { x++ })
	call27(func() { x++ })
	printnl()
}

// but defer does escape to later execution in the function

func f27defer(b bool) {
	x := 0
	if b {
		defer call27(func() { x++ }) // ERROR "stack object .autotmp_[0-9]+ struct \{"
	}
	defer call27(func() { x++ }) // ERROR "stack object .autotmp_[0-9]+ struct \{"
	printnl()                    // ERROR "live at call to printnl: .autotmp_[0-9]+ .autotmp_[0-9]+"
	return                       // ERROR "live at indirect call: .autotmp_[0-9]+"
}

// and newproc (go) escapes to the heap

func f27go(b bool) {
	x := 0
	if b {
		go call27(func() { x++ }) // ERROR "live at call to newobject: &x$" "live at call to newobject: &x .autotmp_[0-9]+$" "live at call to newproc: &x$" // allocate two closures, the func literal, and the wrapper for go
	}
	go call27(func() { x++ }) // ERROR "live at call to newobject: &x$" "live at call to newobject: .autotmp_[0-9]+$" // allocate two closures, the func literal, and the wrapper for go
	printnl()
}

//go:noescape
func call27(func())

// concatstring slice should die on return

var s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 string

func f28(b bool) {
	if b {
		printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10) // ERROR "stack object .autotmp_[0-9]+ \[10\]string$"
	}
	printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10)
	printstring(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10)
}

// map iterator should die on end of range loop

func f29(b bool) {
	if b {
		for k := range m { // ERROR "live at call to (mapiterinit|mapIterStart): .autotmp_[0-9]+$" "live at call to (mapiternext|mapIterNext): .autotmp_[0-9]+$" "stack object .autotmp_[0-9]+ (runtime.hiter|internal/runtime/maps.Iter)$"
			printstring(k) // ERROR "live at call to printstring: .autotmp_[0-9]+$"
		}
	}
	for k := range m { // ERROR "live at call to (mapiterinit|mapIterStart): .autotmp_[0-9]+$" "live at call to (mapiternext|mapIterNext): .autotmp_[0-9]+$"
		printstring(k) // ERROR "live at call to printstring: .autotmp_[0-9]+$"
	}
	for k := range m { // ERROR "live at call to (mapiterinit|mapIterStart): .autotmp_[0-9]+$" "live at call to (mapiternext|mapIterNext): .autotmp_[0-9]+$"
		printstring(k) // ERROR "live at call to printstring: .autotmp_[0-9]+$"
	}
}

// copy of array of pointers should die at end of range loop
var pstructarr [10]pstruct

// Struct size chosen to make pointer to element in pstructarr
// not computable by strength reduction.
type pstruct struct {
	intp *int
	_    [8]byte
}

func f30(b bool) {
	// live temp during printintpointer(p):
	// the internal iterator pointer if a pointer to pstruct in pstructarr
	// can not be easily computed by strength reduction.
	if b {
		for _, p := range pstructarr { // ERROR "stack object .autotmp_[0-9]+ \[10\]pstruct$"
			printintpointer(p.intp) // ERROR "live at call to printintpointer: .autotmp_[0-9]+$"
		}
	}
	for _, p := range pstructarr {
		printintpointer(p.intp) // ERROR "live at call to printintpointer: .autotmp_[0-9]+$"
	}
	for _, p := range pstructarr {
		printintpointer(p.intp) // ERROR "live at call to printintpointer: .autotmp_[0-9]+$"
	}
}

// conversion to interface should not leave temporary behind

func f31(b1, b2, b3 bool) {
	if b1 {
		g31(g18()) // ERROR "stack object .autotmp_[0-9]+ \[2\]string$"
	}
	if b2 {
		h31(g18()) // ERROR "live at call to convT: .autotmp_[0-9]+$" "live at call to newobject: .autotmp_[0-9]+$"
	}
	if b3 {
		panic(g18())
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
		call32(t32.Inc) // ERROR "stack object .autotmp_[0-9]+ struct \{"
	}
	call32(t32.Inc)
	call32(t32.Inc)
}

//go:noescape
func call32(func())

// temporaries introduced during if conditions and && || expressions
// should die once the condition has been acted upon.

var m33 map[interface{}]int

func f33() {
	if m33[byteptr()] == 0 { // ERROR "stack object .autotmp_[0-9]+ interface \{\}$"
		printnl()
		return
	} else {
		printnl()
	}
	printnl()
}

func f34() {
	if m33[byteptr()] == 0 { // ERROR "stack object .autotmp_[0-9]+ interface \{\}$"
		printnl()
		return
	}
	printnl()
}

func f35() {
	if m33[byteptr()] == 0 && // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		m33[byteptr()] == 0 { // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		printnl()
		return
	}
	printnl()
}

func f36() {
	if m33[byteptr()] == 0 || // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		m33[byteptr()] == 0 { // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		printnl()
		return
	}
	printnl()
}

func f37() {
	if (m33[byteptr()] == 0 || // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		m33[byteptr()] == 0) && // ERROR "stack object .autotmp_[0-9]+ interface \{\}"
		m33[byteptr()] == 0 {
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
		select { // ERROR "live at call to selectgo:( .autotmp_[0-9]+)+$" "stack object .autotmp_[0-9]+ \[4\]runtime.scase$"
		case <-fc38():
			printnl()
		case fc38() <- *fi38(1): // ERROR "live at call to fc38:( .autotmp_[0-9]+)+$" "live at call to fi38:( .autotmp_[0-9]+)+$" "stack object .autotmp_[0-9]+ string$"
			printnl()
		case *fi38(2) = <-fc38(): // ERROR "live at call to fc38:( .autotmp_[0-9]+)+$" "live at call to fi38:( .autotmp_[0-9]+)+$" "stack object .autotmp_[0-9]+ string$"
			printnl()
		case *fi38(3), *fb38() = <-fc38(): // ERROR "stack object .autotmp_[0-9]+ string$" "live at call to f[ibc]38:( .autotmp_[0-9]+)+$"
			printnl()
		}
		printnl()
	}
	printnl()
}

// issue 8097: mishandling of x = x during return.

func f39() (x []int) {
	x = []int{1}
	printnl() // ERROR "live at call to printnl: .autotmp_[0-9]+$"
	return x
}

func f39a() (x []int) {
	x = []int{1}
	printnl() // ERROR "live at call to printnl: .autotmp_[0-9]+$"
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

//go:noescape
func useT40(*T40)

func newT40() *T40 {
	ret := T40{}
	ret.m = make(map[int]int, 42) // ERROR "live at call to makemap: &ret$"
	return &ret
}

func good40() {
	ret := T40{}              // ERROR "stack object ret T40$"
	ret.m = make(map[int]int) // ERROR "live at call to rand(32)?: .autotmp_[0-9]+$" "stack object .autotmp_[0-9]+ (runtime.hmap|internal/runtime/maps.Map)$"
	t := &ret
	printnl() // ERROR "live at call to printnl: ret$"
	// Note: ret is live at the printnl because the compiler moves &ret
	// from before the printnl to after.
	useT40(t)
}

func bad40() {
	t := newT40()
	_ = t
	printnl()
}

func ddd1(x, y *int) { // ERROR "live at entry to ddd1: x y$"
	ddd2(x, y) // ERROR "stack object .autotmp_[0-9]+ \[2\]\*int$"
	printnl()
	// Note: no .?autotmp live at printnl.  See issue 16996.
}
func ddd2(a ...*int) { // ERROR "live at entry to ddd2: a$"
	sink = a[0]
}

// issue 16016: autogenerated wrapper should have arguments live
type T struct{}

func (*T) Foo(ptr *int) {}

type R struct{ *T }

// issue 18860: output arguments must be live all the time if there is a defer.
// In particular, at printint r must be live.
func f41(p, q *int) (r *int) { // ERROR "live at entry to f41: p q$"
	r = p
	defer func() {
		recover()
	}()
	printint(0) // ERROR "live at call to printint: .autotmp_[0-9]+ q r$"
	r = q
	return // ERROR "live at call to f41.func1: .autotmp_[0-9]+ r$"
}

func f42() {
	var p, q, r int
	f43([]*int{&p, &q, &r}) // ERROR "stack object .autotmp_[0-9]+ \[3\]\*int$"
	f43([]*int{&p, &r, &q})
	f43([]*int{&q, &p, &r})
}

//go:noescape
func f43(a []*int)

// Assigning to a sub-element that makes up an entire local variable
// should clobber that variable.
func f44(f func() [2]*int) interface{} { // ERROR "live at entry to f44: f"
	type T struct {
		s [1][2]*int
	}
	ret := T{} // ERROR "stack object ret T"
	ret.s[0] = f()
	return ret
}

func f45(a, b, c, d, e, f, g, h, i, j, k, l *byte) { // ERROR "live at entry to f45: a b c d e f g h i j k l"
	f46(a, b, c, d, e, f, g, h, i, j, k, l) // ERROR "live at call to f46: a b c d e f g h i j k l"
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(c)
	runtime.KeepAlive(d)
	runtime.KeepAlive(e)
	runtime.KeepAlive(f)
	runtime.KeepAlive(g)
	runtime.KeepAlive(h)
	runtime.KeepAlive(i)
	runtime.KeepAlive(j)
	runtime.KeepAlive(k)
	runtime.KeepAlive(l)
}

//go:noinline
func f46(a, b, c, d, e, f, g, h, i, j, k, l *byte) {
}
