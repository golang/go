// This interpreter test is designed to run very quickly yet provide
// some coverage of a broad selection of constructs.
//
// Validate this file with 'go run' after editing.
// TODO(adonovan): break this into small files organized by theme.

package main

import (
	"fmt"
	"reflect"
	"strings"
)

func init() {
	// Call of variadic function with (implicit) empty slice.
	if x := fmt.Sprint(); x != "" {
		panic(x)
	}
}

type empty interface{}

type I interface {
	f() int
}

type T struct{ z int }

func (t T) f() int { return t.z }

func use(interface{}) {}

var counter = 2

// Test initialization, including init blocks containing 'return'.
// Assertion is in main.
func init() {
	counter *= 3
	return
	counter *= 3
}

func init() {
	counter *= 5
	return
	counter *= 5
}

// Recursion.
func fib(x int) int {
	if x < 2 {
		return x
	}
	return fib(x-1) + fib(x-2)
}

func fibgen(ch chan int) {
	for x := 0; x < 10; x++ {
		ch <- fib(x)
	}
	close(ch)
}

// Goroutines and channels.
func init() {
	ch := make(chan int)
	go fibgen(ch)
	var fibs []int
	for v := range ch {
		fibs = append(fibs, v)
		if len(fibs) == 10 {
			break
		}
	}
	if x := fmt.Sprint(fibs); x != "[0 1 1 2 3 5 8 13 21 34]" {
		panic(x)
	}
}

// Test of aliasing.
func init() {
	type S struct {
		a, b string
	}

	s1 := []string{"foo", "bar"}
	s2 := s1 // creates an alias
	s2[0] = "wiz"
	if x := fmt.Sprint(s1, s2); x != "[wiz bar] [wiz bar]" {
		panic(x)
	}

	pa1 := &[2]string{"foo", "bar"}
	pa2 := pa1 // creates an alias
	pa2[0] = "wiz"
	if x := fmt.Sprint(*pa1, *pa2); x != "[wiz bar] [wiz bar]" {
		panic(x)
	}

	a1 := [2]string{"foo", "bar"}
	a2 := a1 // creates a copy
	a2[0] = "wiz"
	if x := fmt.Sprint(a1, a2); x != "[foo bar] [wiz bar]" {
		panic(x)
	}

	t1 := S{"foo", "bar"}
	t2 := t1 // copy
	t2.a = "wiz"
	if x := fmt.Sprint(t1, t2); x != "{foo bar} {wiz bar}" {
		panic(x)
	}
}

func main() {
	print() // legal

	if counter != 2*3*5 {
		panic(counter)
	}

	// Test builtins (e.g. complex) preserve named argument types.
	type N complex128
	var n N
	n = complex(1.0, 2.0)
	if n != complex(1.0, 2.0) {
		panic(n)
	}
	if x := reflect.TypeOf(n).String(); x != "main.N" {
		panic(x)
	}
	if real(n) != 1.0 || imag(n) != 2.0 {
		panic(n)
	}

	// Channel + select.
	ch := make(chan int, 1)
	select {
	case ch <- 1:
		// ok
	default:
		panic("couldn't send")
	}
	if <-ch != 1 {
		panic("couldn't receive")
	}
	// A "receive" select-case that doesn't declare its vars.  (regression test)
	anint := 0
	ok := false
	select {
	case anint, ok = <-ch:
	case anint = <-ch:
	default:
	}
	_ = anint
	_ = ok

	// Anon structs with methods.
	anon := struct{ T }{T: T{z: 1}}
	if x := anon.f(); x != 1 {
		panic(x)
	}
	var i I = anon
	if x := i.f(); x != 1 {
		panic(x)
	}
	// NB. precise output of reflect.Type.String is undefined.
	if x := reflect.TypeOf(i).String(); x != "struct { main.T }" && x != "struct{main.T}" {
		panic(x)
	}

	// fmt.
	const message = "Hello, World!"
	if fmt.Sprint("Hello", ", ", "World", "!") != message {
		panic("oops")
	}

	// Type assertion.
	type S struct {
		f int
	}
	var e empty = S{f: 42}
	switch v := e.(type) {
	case S:
		if v.f != 42 {
			panic(v.f)
		}
	default:
		panic(reflect.TypeOf(v))
	}
	if i, ok := e.(I); ok {
		panic(i)
	}

	// Switch.
	var x int
	switch x {
	case 1:
		panic(x)
		fallthrough
	case 2, 3:
		panic(x)
	default:
		// ok
	}
	// empty switch
	switch {
	}
	// empty switch
	switch {
	default:
	}
	// empty switch
	switch {
	default:
		fallthrough
	case false:
	}

	// string -> []rune conversion.
	use([]rune("foo"))

	// Calls of form x.f().
	type S2 struct {
		f func() int
	}
	S2{f: func() int { return 1 }}.f() // field is a func value
	T{}.f()                            // method call
	i.f()                              // interface method invocation
	(interface {
		f() int
	}(T{})).f() // anon interface method invocation

	// Map lookup.
	if v, ok := map[string]string{}["foo5"]; v != "" || ok {
		panic("oops")
	}

	// Regression test: implicit address-taken struct literal
	// inside literal map element.
	_ = map[int]*struct{}{0: {}}
}

type mybool bool

func (mybool) f() {}

func init() {
	type mybool bool
	var b mybool
	var i interface{} = b || b // result preserves types of operands
	_ = i.(mybool)

	i = false && b // result preserves type of "typed" operand
	_ = i.(mybool)

	i = b || true // result preserves type of "typed" operand
	_ = i.(mybool)
}

func init() {
	var x, y int
	var b mybool = x == y // x==y is an untyped bool
	b.f()
}

// Simple closures.
func init() {
	b := 3
	f := func(a int) int {
		return a + b
	}
	b++
	if x := f(1); x != 5 { // 1+4 == 5
		panic(x)
	}
	b++
	if x := f(2); x != 7 { // 2+5 == 7
		panic(x)
	}
	if b := f(1) < 16 || f(2) < 17; !b {
		panic("oops")
	}
}

// Shifts.
func init() {
	var i int64 = 1
	var u uint64 = 1 << 32
	if x := i << uint32(u); x != 1 {
		panic(x)
	}
	if x := i << uint64(u); x != 0 {
		panic(x)
	}
}

// Implicit conversion of delete() key operand.
func init() {
	type I interface{}
	m := make(map[I]bool)
	m[1] = true
	m[I(2)] = true
	if len(m) != 2 {
		panic(m)
	}
	delete(m, I(1))
	delete(m, 2)
	if len(m) != 0 {
		panic(m)
	}
}

// An I->I conversion always succeeds.
func init() {
	var x I
	if I(x) != I(nil) {
		panic("I->I conversion failed")
	}
}

// An I->I type-assert fails iff the value is nil.
func init() {
	defer func() {
		r := fmt.Sprint(recover())
		// Exact error varies by toolchain.
		if r != "runtime error: interface conversion: interface is nil, not main.I" &&
			r != "interface conversion: interface is nil, not main.I" {
			panic("I->I type assertion succeeded for nil value")
		}
	}()
	var x I
	_ = x.(I)
}

//////////////////////////////////////////////////////////////////////
// Variadic bridge methods and interface thunks.

type VT int

var vcount = 0

func (VT) f(x int, y ...string) {
	vcount++
	if x != 1 {
		panic(x)
	}
	if len(y) != 2 || y[0] != "foo" || y[1] != "bar" {
		panic(y)
	}
}

type VS struct {
	VT
}

type VI interface {
	f(x int, y ...string)
}

func init() {
	foobar := []string{"foo", "bar"}
	var s VS
	s.f(1, "foo", "bar")
	s.f(1, foobar...)
	if vcount != 2 {
		panic("s.f not called twice")
	}

	fn := VI.f
	fn(s, 1, "foo", "bar")
	fn(s, 1, foobar...)
	if vcount != 4 {
		panic("I.f not called twice")
	}
}

// Multiple labels on same statement.
func multipleLabels() {
	var trace []int
	i := 0
one:
two:
	for ; i < 3; i++ {
		trace = append(trace, i)
		switch i {
		case 0:
			continue two
		case 1:
			i++
			goto one
		case 2:
			break two
		}
	}
	if x := fmt.Sprint(trace); x != "[0 1 2]" {
		panic(x)
	}
}

func init() {
	multipleLabels()
}

func init() {
	// Struct equivalence ignores blank fields.
	type s struct{ x, _, z int }
	s1 := s{x: 1, z: 3}
	s2 := s{x: 1, z: 3}
	if s1 != s2 {
		panic("not equal")
	}
}

func init() {
	// A slice var can be compared to const []T nil.
	var i interface{} = []string{"foo"}
	var j interface{} = []string(nil)
	if i.([]string) == nil {
		panic("expected i non-nil")
	}
	if j.([]string) != nil {
		panic("expected j nil")
	}
	// But two slices cannot be compared, even if one is nil.
	defer func() {
		r := fmt.Sprint(recover())
		if !(strings.Contains(r, "compar") && strings.Contains(r, "[]string")) {
			panic("want panic from slice comparison, got " + r)
		}
	}()
	_ = i == j // interface comparison recurses on types
}

func init() {
	// Regression test for SSA renaming bug.
	var ints []int
	for range "foo" {
		var x int
		x++
		ints = append(ints, x)
	}
	if fmt.Sprint(ints) != "[1 1 1]" {
		panic(ints)
	}
}

// Regression test for issue 6949:
// []byte("foo") is not a constant since it allocates memory.
func init() {
	var r string
	for i, b := range "ABC" {
		x := []byte("abc")
		x[i] = byte(b)
		r += string(x)
	}
	if r != "AbcaBcabC" {
		panic(r)
	}
}

// Test of 3-operand x[lo:hi:max] slice.
func init() {
	s := []int{0, 1, 2, 3}
	lenCapLoHi := func(x []int) [4]int { return [4]int{len(x), cap(x), x[0], x[len(x)-1]} }
	if got := lenCapLoHi(s[1:3]); got != [4]int{2, 3, 1, 2} {
		panic(got)
	}
	if got := lenCapLoHi(s[1:3:3]); got != [4]int{2, 2, 1, 2} {
		panic(got)
	}
	max := 3
	if "a"[0] == 'a' {
		max = 2 // max is non-constant, even in SSA form
	}
	if got := lenCapLoHi(s[1:2:max]); got != [4]int{1, 1, 1, 1} {
		panic(got)
	}
}

var one = 1 // not a constant

// Test makeslice.
func init() {
	check := func(s []string, wantLen, wantCap int) {
		if len(s) != wantLen {
			panic(len(s))
		}
		if cap(s) != wantCap {
			panic(cap(s))
		}
	}
	//                                       SSA form:
	check(make([]string, 10), 10, 10)     // new([10]string)[:10]
	check(make([]string, one), 1, 1)      // make([]string, one, one)
	check(make([]string, 0, 10), 0, 10)   // new([10]string)[:0]
	check(make([]string, 0, one), 0, 1)   // make([]string, 0, one)
	check(make([]string, one, 10), 1, 10) // new([10]string)[:one]
	check(make([]string, one, one), 1, 1) // make([]string, one, one)
}

// Test that a nice error is issued by indirection wrappers.
func init() {
	var ptr *T
	var i I = ptr

	defer func() {
		r := fmt.Sprint(recover())
		// Exact error varies by toolchain:
		if r != "runtime error: value method (main.T).f called using nil *main.T pointer" &&
			r != "value method (main.T).f called using nil *main.T pointer" {
			panic("want panic from call with nil receiver, got " + r)
		}
	}()
	i.f()
	panic("unreachable")
}

// Regression test for a subtle bug in which copying values would causes
// subcomponents of aggregate variables to change address, breaking
// aliases.
func init() {
	type T struct{ f int }
	var x T
	p := &x.f
	x = T{}
	*p = 1
	if x.f != 1 {
		panic("lost store")
	}
	if p != &x.f {
		panic("unstable address")
	}
}
