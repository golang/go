// errorcheck -0 -N -m -l

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is disabled.
// Registerization is disabled too (-N), which should
// have no effect on escape analysis.

package foo

import (
	"fmt"
	"unsafe"
)

var gxx *int

func foo1(x int) { // ERROR "moved to heap: x$"
	gxx = &x // ERROR "&x escapes to heap$"
}

func foo2(yy *int) { // ERROR "leaking param: yy$"
	gxx = yy
}

func foo3(x int) *int { // ERROR "moved to heap: x$"
	return &x // ERROR "&x escapes to heap$"
}

type T *T

func foo3b(t T) { // ERROR "leaking param: t$"
	*t = t
}

// xx isn't going anywhere, so use of yy is ok
func foo4(xx, yy *int) { // ERROR "foo4 xx does not escape$" "foo4 yy does not escape$"
	xx = yy
}

// xx isn't going anywhere, so taking address of yy is ok
func foo5(xx **int, yy *int) { // ERROR "foo5 xx does not escape$" "foo5 yy does not escape$"
	xx = &yy // ERROR "foo5 &yy does not escape$"
}

func foo6(xx **int, yy *int) { // ERROR "foo6 xx does not escape$" "leaking param: yy$"
	*xx = yy
}

func foo7(xx **int, yy *int) { // ERROR "foo7 xx does not escape$" "foo7 yy does not escape$"
	**xx = *yy
}

func foo8(xx, yy *int) int { // ERROR "foo8 xx does not escape$" "foo8 yy does not escape$"
	xx = yy
	return *xx
}

func foo9(xx, yy *int) *int { // ERROR "leaking param: xx to result ~r2 level=0$" "leaking param: yy to result ~r2 level=0$"
	xx = yy
	return xx
}

func foo10(xx, yy *int) { // ERROR "foo10 xx does not escape$" "foo10 yy does not escape$"
	*xx = *yy
}

func foo11() int {
	x, y := 0, 42
	xx := &x // ERROR "foo11 &x does not escape$"
	yy := &y // ERROR "foo11 &y does not escape$"
	*xx = *yy
	return x
}

var xxx **int

func foo12(yyy **int) { // ERROR "leaking param: yyy$"
	xxx = yyy
}

// Must treat yyy as leaking because *yyy leaks, and the escape analysis
// summaries in exported metadata do not distinguish these two cases.
func foo13(yyy **int) { // ERROR "leaking param content: yyy$"
	*xxx = *yyy
}

func foo14(yyy **int) { // ERROR "foo14 yyy does not escape$"
	**xxx = **yyy
}

func foo15(yy *int) { // ERROR "moved to heap: yy$"
	xxx = &yy // ERROR "&yy escapes to heap$"
}

func foo16(yy *int) { // ERROR "leaking param: yy$"
	*xxx = yy
}

func foo17(yy *int) { // ERROR "foo17 yy does not escape$"
	**xxx = *yy
}

func foo18(y int) { // ERROR "moved to heap: y$"
	*xxx = &y // ERROR "&y escapes to heap$"
}

func foo19(y int) {
	**xxx = y
}

type Bar struct {
	i  int
	ii *int
}

func NewBar() *Bar {
	return &Bar{42, nil} // ERROR "&Bar literal escapes to heap$"
}

func NewBarp(x *int) *Bar { // ERROR "leaking param: x to result ~r1 level=-1$"
	return &Bar{42, x} // ERROR "&Bar literal escapes to heap$"
}

func NewBarp2(x *int) *Bar { // ERROR "NewBarp2 x does not escape$"
	return &Bar{*x, nil} // ERROR "&Bar literal escapes to heap$"
}

func (b *Bar) NoLeak() int { // ERROR "\(\*Bar\).NoLeak b does not escape$"
	return *(b.ii)
}

func (b *Bar) Leak() *int { // ERROR "leaking param: b to result ~r0 level=0$"
	return &b.i // ERROR "&b.i escapes to heap$"
}

func (b *Bar) AlsoNoLeak() *int { // ERROR "leaking param: b to result ~r0 level=1$"
	return b.ii
}

func (b Bar) AlsoLeak() *int { // ERROR "leaking param: b to result ~r0 level=0$"
	return b.ii
}

func (b Bar) LeaksToo() *int { // ERROR "leaking param: b to result ~r0 level=0$"
	v := 0    // ERROR "moved to heap: v$"
	b.ii = &v // ERROR "&v escapes to heap$"
	return b.ii
}

func (b *Bar) LeaksABit() *int { // ERROR "leaking param: b to result ~r0 level=1$"
	v := 0    // ERROR "moved to heap: v$"
	b.ii = &v // ERROR "&v escapes to heap$"
	return b.ii
}

func (b Bar) StillNoLeak() int { // ERROR "Bar.StillNoLeak b does not escape$"
	v := 0
	b.ii = &v // ERROR "Bar.StillNoLeak &v does not escape$"
	return b.i
}

func goLeak(b *Bar) { // ERROR "leaking param: b$"
	go b.NoLeak()
}

type Bar2 struct {
	i  [12]int
	ii []int
}

func NewBar2() *Bar2 {
	return &Bar2{[12]int{42}, nil} // ERROR "&Bar2 literal escapes to heap$"
}

func (b *Bar2) NoLeak() int { // ERROR "\(\*Bar2\).NoLeak b does not escape$"
	return b.i[0]
}

func (b *Bar2) Leak() []int { // ERROR "leaking param: b to result ~r0 level=0$"
	return b.i[:] // ERROR "b.i escapes to heap$"
}

func (b *Bar2) AlsoNoLeak() []int { // ERROR "leaking param: b to result ~r0 level=1$"
	return b.ii[0:1]
}

func (b Bar2) AgainNoLeak() [12]int { // ERROR "Bar2.AgainNoLeak b does not escape$"
	return b.i
}

func (b *Bar2) LeakSelf() { // ERROR "leaking param: b$"
	b.ii = b.i[0:4] // ERROR "b.i escapes to heap$"
}

func (b *Bar2) LeakSelf2() { // ERROR "leaking param: b$"
	var buf []int
	buf = b.i[0:] // ERROR "b.i escapes to heap$"
	b.ii = buf
}

func foo21() func() int {
	x := 42
	return func() int { // ERROR "func literal escapes to heap$"
		return x
	}
}

func foo21a() func() int {
	x := 42             // ERROR "moved to heap: x$"
	return func() int { // ERROR "func literal escapes to heap$"
		x++ // ERROR "&x escapes to heap$"
		return x
	}
}

func foo22() int {
	x := 42
	return func() int { // ERROR "foo22 func literal does not escape$"
		return x
	}()
}

func foo23(x int) func() int {
	return func() int { // ERROR "func literal escapes to heap$"
		return x
	}
}

func foo23a(x int) func() int {
	f := func() int { // ERROR "func literal escapes to heap$"
		return x
	}
	return f
}

func foo23b(x int) *(func() int) {
	f := func() int { return x } // ERROR "func literal escapes to heap$" "moved to heap: f$"
	return &f                    // ERROR "&f escapes to heap$"
}

func foo23c(x int) func() int { // ERROR "moved to heap: x$"
	return func() int { // ERROR "func literal escapes to heap$"
		x++ // ERROR "&x escapes to heap$"
		return x
	}
}

func foo24(x int) int {
	return func() int { // ERROR "foo24 func literal does not escape$"
		return x
	}()
}

var x *int

func fooleak(xx *int) int { // ERROR "leaking param: xx$"
	x = xx
	return *x
}

func foonoleak(xx *int) int { // ERROR "foonoleak xx does not escape$"
	return *x + *xx
}

func foo31(x int) int { // ERROR "moved to heap: x$"
	return fooleak(&x) // ERROR "&x escapes to heap$"
}

func foo32(x int) int {
	return foonoleak(&x) // ERROR "foo32 &x does not escape$"
}

type Foo struct {
	xx *int
	x  int
}

var F Foo
var pf *Foo

func (f *Foo) fooleak() { // ERROR "leaking param: f$"
	pf = f
}

func (f *Foo) foonoleak() { // ERROR "\(\*Foo\).foonoleak f does not escape$"
	F.x = f.x
}

func (f *Foo) Leak() { // ERROR "leaking param: f$"
	f.fooleak()
}

func (f *Foo) NoLeak() { // ERROR "\(\*Foo\).NoLeak f does not escape$"
	f.foonoleak()
}

func foo41(x int) { // ERROR "moved to heap: x$"
	F.xx = &x // ERROR "&x escapes to heap$"
}

func (f *Foo) foo42(x int) { // ERROR "\(\*Foo\).foo42 f does not escape$" "moved to heap: x$"
	f.xx = &x // ERROR "&x escapes to heap$"
}

func foo43(f *Foo, x int) { // ERROR "foo43 f does not escape$" "moved to heap: x$"
	f.xx = &x // ERROR "&x escapes to heap$"
}

func foo44(yy *int) { // ERROR "leaking param: yy$"
	F.xx = yy
}

func (f *Foo) foo45() { // ERROR "\(\*Foo\).foo45 f does not escape$"
	F.x = f.x
}

// See foo13 above for explanation of why f leaks.
func (f *Foo) foo46() { // ERROR "leaking param content: f$"
	F.xx = f.xx
}

func (f *Foo) foo47() { // ERROR "leaking param: f$"
	f.xx = &f.x // ERROR "&f.x escapes to heap$"
}

var ptrSlice []*int

func foo50(i *int) { // ERROR "leaking param: i$"
	ptrSlice[0] = i
}

var ptrMap map[*int]*int

func foo51(i *int) { // ERROR "leaking param: i$"
	ptrMap[i] = i
}

func indaddr1(x int) *int { // ERROR "moved to heap: x$"
	return &x // ERROR "&x escapes to heap$"
}

func indaddr2(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	return *&x // ERROR "indaddr2 &x does not escape$"
}

func indaddr3(x *int32) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	return *(**int)(unsafe.Pointer(&x)) // ERROR "indaddr3 &x does not escape$"
}

// From package math:

func Float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f)) // ERROR "Float32bits &f does not escape$"
}

func Float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b)) // ERROR "Float32frombits &b does not escape$"
}

func Float64bits(f float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&f)) // ERROR "Float64bits &f does not escape$"
}

func Float64frombits(b uint64) float64 {
	return *(*float64)(unsafe.Pointer(&b)) // ERROR "Float64frombits &b does not escape$"
}

// contrast with
func float64bitsptr(f float64) *uint64 { // ERROR "moved to heap: f$"
	return (*uint64)(unsafe.Pointer(&f)) // ERROR "&f escapes to heap$"
}

func float64ptrbitsptr(f *float64) *uint64 { // ERROR "leaking param: f to result ~r1 level=0$"
	return (*uint64)(unsafe.Pointer(f))
}

func typesw(i interface{}) *int { // ERROR "leaking param: i to result ~r1 level=0$"
	switch val := i.(type) {
	case *int:
		return val
	case *int8:
		v := int(*val) // ERROR "moved to heap: v$"
		return &v      // ERROR "&v escapes to heap$"
	}
	return nil
}

func exprsw(i *int) *int { // ERROR "leaking param: i to result ~r1 level=0$"
	switch j := i; *j + 110 {
	case 12:
		return j
	case 42:
		return nil
	}
	return nil

}

// assigning to an array element is like assigning to the array
func foo60(i *int) *int { // ERROR "leaking param: i to result ~r1 level=0$"
	var a [12]*int
	a[0] = i
	return a[1]
}

func foo60a(i *int) *int { // ERROR "foo60a i does not escape$"
	var a [12]*int
	a[0] = i
	return nil
}

// assigning to a struct field  is like assigning to the struct
func foo61(i *int) *int { // ERROR "leaking param: i to result ~r1 level=0$"
	type S struct {
		a, b *int
	}
	var s S
	s.a = i
	return s.b
}

func foo61a(i *int) *int { // ERROR "foo61a i does not escape$"
	type S struct {
		a, b *int
	}
	var s S
	s.a = i
	return nil
}

// assigning to a struct field is like assigning to the struct but
// here this subtlety is lost, since s.a counts as an assignment to a
// track-losing dereference.
func foo62(i *int) *int { // ERROR "leaking param: i$"
	type S struct {
		a, b *int
	}
	s := new(S) // ERROR "foo62 new\(S\) does not escape$"
	s.a = i
	return nil // s.b
}

type M interface {
	M()
}

func foo63(m M) { // ERROR "foo63 m does not escape$"
}

func foo64(m M) { // ERROR "leaking param: m$"
	m.M()
}

func foo64b(m M) { // ERROR "leaking param: m$"
	defer m.M()
}

type MV int

func (MV) M() {}

func foo65() {
	var mv MV
	foo63(&mv) // ERROR "foo65 &mv does not escape$"
}

func foo66() {
	var mv MV  // ERROR "moved to heap: mv$"
	foo64(&mv) // ERROR "&mv escapes to heap$"
}

func foo67() {
	var mv MV
	foo63(mv) // ERROR "foo67 mv does not escape$"
}

func foo68() {
	var mv MV
	// escapes but it's an int so irrelevant
	foo64(mv) // ERROR "mv escapes to heap$"
}

func foo69(m M) { // ERROR "leaking param: m$"
	foo64(m)
}

func foo70(mv1 *MV, m M) { // ERROR "leaking param: m$" "leaking param: mv1$"
	m = mv1 // ERROR "mv1 escapes to heap$"
	foo64(m)
}

func foo71(x *int) []*int { // ERROR "leaking param: x$"
	var y []*int
	y = append(y, x)
	return y
}

func foo71a(x int) []*int { // ERROR "moved to heap: x$"
	var y []*int
	y = append(y, &x) // ERROR "&x escapes to heap$"
	return y
}

func foo72() {
	var x int
	var y [1]*int
	y[0] = &x // ERROR "foo72 &x does not escape$"
}

func foo72aa() [10]*int {
	var x int // ERROR "moved to heap: x$"
	var y [10]*int
	y[0] = &x // ERROR "&x escapes to heap$"
	return y
}

func foo72a() {
	var y [10]*int
	for i := 0; i < 10; i++ {
		// escapes its scope
		x := i    // ERROR "moved to heap: x$"
		y[i] = &x // ERROR "&x escapes to heap$"
	}
	return
}

func foo72b() [10]*int {
	var y [10]*int
	for i := 0; i < 10; i++ {
		x := i    // ERROR "moved to heap: x$"
		y[i] = &x // ERROR "&x escapes to heap$"
	}
	return y
}

// issue 2145
func foo73() {
	s := []int{3, 2, 1} // ERROR "foo73 \[\]int literal does not escape$"
	for _, v := range s {
		vv := v
		// actually just escapes its scope
		defer func() { // ERROR "func literal escapes to heap$"
			println(vv)
		}()
	}
}

func foo731() {
	s := []int{3, 2, 1} // ERROR "foo731 \[\]int literal does not escape$"
	for _, v := range s {
		vv := v // ERROR "moved to heap: vv$"
		// actually just escapes its scope
		defer func() { // ERROR "func literal escapes to heap$"
			vv = 42 // ERROR "&vv escapes to heap$"
			println(vv)
		}()
	}
}

func foo74() {
	s := []int{3, 2, 1} // ERROR "foo74 \[\]int literal does not escape$"
	for _, v := range s {
		vv := v
		// actually just escapes its scope
		fn := func() { // ERROR "func literal escapes to heap$"
			println(vv)
		}
		defer fn()
	}
}

func foo74a() {
	s := []int{3, 2, 1} // ERROR "foo74a \[\]int literal does not escape$"
	for _, v := range s {
		vv := v // ERROR "moved to heap: vv$"
		// actually just escapes its scope
		fn := func() { // ERROR "func literal escapes to heap$"
			vv += 1 // ERROR "&vv escapes to heap$"
			println(vv)
		}
		defer fn()
	}
}

// issue 3975
func foo74b() {
	var array [3]func()
	s := []int{3, 2, 1} // ERROR "foo74b \[\]int literal does not escape$"
	for i, v := range s {
		vv := v
		// actually just escapes its scope
		array[i] = func() { // ERROR "func literal escapes to heap$"
			println(vv)
		}
	}
}

func foo74c() {
	var array [3]func()
	s := []int{3, 2, 1} // ERROR "foo74c \[\]int literal does not escape$"
	for i, v := range s {
		vv := v // ERROR "moved to heap: vv$"
		// actually just escapes its scope
		array[i] = func() { // ERROR "func literal escapes to heap$"
			println(&vv) // ERROR "&vv escapes to heap$" "foo74c.func1 &vv does not escape$"
		}
	}
}

func myprint(y *int, x ...interface{}) *int { // ERROR "leaking param: y to result ~r2 level=0$" "myprint x does not escape$"
	return y
}

func myprint1(y *int, x ...interface{}) *interface{} { // ERROR "leaking param: x to result ~r2 level=0$" "myprint1 y does not escape$"
	return &x[0] // ERROR "&x\[0\] escapes to heap$"
}

func foo75(z *int) { // ERROR "foo75 z does not escape$"
	myprint(z, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo75 ... argument does not escape$"
}

func foo75a(z *int) { // ERROR "foo75a z does not escape$"
	myprint1(z, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo75a ... argument does not escape$"
}

func foo75esc(z *int) { // ERROR "leaking param: z$"
	gxx = myprint(z, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo75esc ... argument does not escape$"
}

func foo75aesc(z *int) { // ERROR "foo75aesc z does not escape$"
	var ppi **interface{}       // assignments to pointer dereferences lose track
	*ppi = myprint1(z, 1, 2, 3) // ERROR "... argument escapes to heap$" "1 escapes to heap$" "2 escapes to heap$" "3 escapes to heap$"
}

func foo75aesc1(z *int) { // ERROR "foo75aesc1 z does not escape$"
	sink = myprint1(z, 1, 2, 3) // ERROR "... argument escapes to heap$" "1 escapes to heap$" "2 escapes to heap$" "3 escapes to heap$" "myprint1\(z, 1, 2, 3\) escapes to heap$"
}

func foo76(z *int) { // ERROR "z does not escape"
	myprint(nil, z) // ERROR "foo76 ... argument does not escape$" "z does not escape"
}

func foo76a(z *int) { // ERROR "z does not escape"
	myprint1(nil, z) // ERROR "foo76a ... argument does not escape$" "z does not escape"
}

func foo76b() {
	myprint(nil, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo76b ... argument does not escape$"
}

func foo76c() {
	myprint1(nil, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo76c ... argument does not escape$"
}

func foo76d() {
	defer myprint(nil, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo76d ... argument does not escape$"
}

func foo76e() {
	defer myprint1(nil, 1, 2, 3) // ERROR "1 does not escape" "2 does not escape" "3 does not escape" "foo76e ... argument does not escape$"
}

func foo76f() {
	for {
		// TODO: This one really only escapes its scope, but we don't distinguish yet.
		defer myprint(nil, 1, 2, 3) // ERROR "... argument escapes to heap$" "1 escapes to heap$" "2 escapes to heap$" "3 escapes to heap$"
	}
}

func foo76g() {
	for {
		defer myprint1(nil, 1, 2, 3) // ERROR "... argument escapes to heap$" "1 escapes to heap$" "2 escapes to heap$" "3 escapes to heap$"
	}
}

func foo77(z []interface{}) { // ERROR "foo77 z does not escape$"
	myprint(nil, z...) // z does not escape
}

func foo77a(z []interface{}) { // ERROR "foo77a z does not escape$"
	myprint1(nil, z...)
}

func foo77b(z []interface{}) { // ERROR "leaking param: z$"
	var ppi **interface{}
	*ppi = myprint1(nil, z...)
}

func foo77c(z []interface{}) { // ERROR "leaking param: z$"
	sink = myprint1(nil, z...) // ERROR "myprint1\(nil, z...\) escapes to heap$"
}

func dotdotdot() {
	i := 0
	myprint(nil, &i) // ERROR "&i does not escape" "dotdotdot ... argument does not escape$"

	j := 0
	myprint1(nil, &j) // ERROR "&j does not escape" "dotdotdot ... argument does not escape$"
}

func foo78(z int) *int { // ERROR "moved to heap: z$"
	return &z // ERROR "&z escapes to heap$"
}

func foo78a(z int) *int { // ERROR "moved to heap: z$"
	y := &z   // ERROR "&z escapes to heap$"
	x := &y   // ERROR "foo78a &y does not escape$"
	return *x // really return y
}

func foo79() *int {
	return new(int) // ERROR "new\(int\) escapes to heap$"
}

func foo80() *int {
	var z *int
	for {
		// Really just escapes its scope but we don't distinguish
		z = new(int) // ERROR "new\(int\) escapes to heap$"
	}
	_ = z
	return nil
}

func foo81() *int {
	for {
		z := new(int) // ERROR "foo81 new\(int\) does not escape$"
		_ = z
	}
	return nil
}

func tee(p *int) (x, y *int) { return p, p } // ERROR "leaking param: p to result x level=0$" "leaking param: p to result y level=0$"

func noop(x, y *int) {} // ERROR "noop x does not escape$" "noop y does not escape$"

func foo82() {
	var x, y, z int  // ERROR "moved to heap: x$" "moved to heap: y$" "moved to heap: z$"
	go noop(tee(&z)) // ERROR "&z escapes to heap$"
	go noop(&x, &y)  // ERROR "&x escapes to heap$" "&y escapes to heap$"
	for {
		var u, v, w int     // ERROR "moved to heap: u$" "moved to heap: v$" "moved to heap: w$"
		defer noop(tee(&u)) // ERROR "&u escapes to heap$"
		defer noop(&v, &w)  // ERROR "&v escapes to heap$" "&w escapes to heap$"
	}
}

type Fooer interface {
	Foo()
}

type LimitedFooer struct {
	Fooer
	N int64
}

func LimitFooer(r Fooer, n int64) Fooer { // ERROR "leaking param: r to result ~r2 level=-1$"
	return &LimitedFooer{r, n} // ERROR "&LimitedFooer literal escapes to heap$"
}

func foo90(x *int) map[*int]*int { // ERROR "leaking param: x$"
	return map[*int]*int{nil: x} // ERROR "map\[\*int\]\*int literal escapes to heap$"
}

func foo91(x *int) map[*int]*int { // ERROR "leaking param: x$"
	return map[*int]*int{x: nil} // ERROR "map\[\*int\]\*int literal escapes to heap$"
}

func foo92(x *int) [2]*int { // ERROR "leaking param: x to result ~r1 level=0$"
	return [2]*int{x, nil}
}

// does not leak c
func foo93(c chan *int) *int { // ERROR "foo93 c does not escape$"
	for v := range c {
		return v
	}
	return nil
}

// does not leak m
func foo94(m map[*int]*int, b bool) *int { // ERROR "leaking param: m to result ~r2 level=1"
	for k, v := range m {
		if b {
			return k
		}
		return v
	}
	return nil
}

// does leak x
func foo95(m map[*int]*int, x *int) { // ERROR "foo95 m does not escape$" "leaking param: x$"
	m[x] = x
}

// does not leak m but does leak content
func foo96(m []*int) *int { // ERROR "leaking param: m to result ~r1 level=1"
	return m[0]
}

// does leak m
func foo97(m [1]*int) *int { // ERROR "leaking param: m to result ~r1 level=0$"
	return m[0]
}

// does not leak m
func foo98(m map[int]*int) *int { // ERROR "foo98 m does not escape$"
	return m[0]
}

// does leak m
func foo99(m *[1]*int) []*int { // ERROR "leaking param: m to result ~r1 level=0$"
	return m[:]
}

// does not leak m
func foo100(m []*int) *int { // ERROR "leaking param: m to result ~r1 level=1"
	for _, v := range m {
		return v
	}
	return nil
}

// does leak m
func foo101(m [1]*int) *int { // ERROR "leaking param: m to result ~r1 level=0$"
	for _, v := range m {
		return v
	}
	return nil
}

// does not leak m
func foo101a(m [1]*int) *int { // ERROR "foo101a m does not escape$"
	for i := range m { // ERROR "moved to heap: i$"
		return &i // ERROR "&i escapes to heap$"
	}
	return nil
}

// does leak x
func foo102(m []*int, x *int) { // ERROR "foo102 m does not escape$" "leaking param: x$"
	m[0] = x
}

// does not leak x
func foo103(m [1]*int, x *int) { // ERROR "foo103 m does not escape$" "foo103 x does not escape$"
	m[0] = x
}

var y []*int

// does not leak x but does leak content
func foo104(x []*int) { // ERROR "leaking param content: x"
	copy(y, x)
}

// does not leak x but does leak content
func foo105(x []*int) { // ERROR "leaking param content: x"
	_ = append(y, x...)
}

// does leak x
func foo106(x *int) { // ERROR "leaking param: x$"
	_ = append(y, x)
}

func foo107(x *int) map[*int]*int { // ERROR "leaking param: x$"
	return map[*int]*int{x: nil} // ERROR "map\[\*int\]\*int literal escapes to heap$"
}

func foo108(x *int) map[*int]*int { // ERROR "leaking param: x$"
	return map[*int]*int{nil: x} // ERROR "map\[\*int\]\*int literal escapes to heap$"
}

func foo109(x *int) *int { // ERROR "leaking param: x$"
	m := map[*int]*int{x: nil} // ERROR "foo109 map\[\*int\]\*int literal does not escape$"
	for k, _ := range m {
		return k
	}
	return nil
}

func foo110(x *int) *int { // ERROR "leaking param: x$"
	m := map[*int]*int{nil: x} // ERROR "foo110 map\[\*int\]\*int literal does not escape$"
	return m[nil]
}

func foo111(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0"
	m := []*int{x} // ERROR "foo111 \[\]\*int literal does not escape$"
	return m[0]
}

func foo112(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	m := [1]*int{x}
	return m[0]
}

func foo113(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	m := Bar{ii: x}
	return m.ii
}

func foo114(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	m := &Bar{ii: x} // ERROR "foo114 &Bar literal does not escape$"
	return m.ii
}

func foo115(x *int) *int { // ERROR "leaking param: x to result ~r1 level=0$"
	return (*int)(unsafe.Pointer(uintptr(unsafe.Pointer(x)) + 1))
}

func foo116(b bool) *int {
	if b {
		x := 1    // ERROR "moved to heap: x$"
		return &x // ERROR "&x escapes to heap$"
	} else {
		y := 1    // ERROR "moved to heap: y$"
		return &y // ERROR "&y escapes to heap$"
	}
	return nil
}

func foo117(unknown func(interface{})) { // ERROR "foo117 unknown does not escape$"
	x := 1      // ERROR "moved to heap: x$"
	unknown(&x) // ERROR "&x escapes to heap$"
}

func foo118(unknown func(*int)) { // ERROR "foo118 unknown does not escape$"
	x := 1      // ERROR "moved to heap: x$"
	unknown(&x) // ERROR "&x escapes to heap$"
}

func external(*int)

func foo119(x *int) { // ERROR "leaking param: x$"
	external(x)
}

func foo120() {
	// formerly exponential time analysis
L1:
L2:
L3:
L4:
L5:
L6:
L7:
L8:
L9:
L10:
L11:
L12:
L13:
L14:
L15:
L16:
L17:
L18:
L19:
L20:
L21:
L22:
L23:
L24:
L25:
L26:
L27:
L28:
L29:
L30:
L31:
L32:
L33:
L34:
L35:
L36:
L37:
L38:
L39:
L40:
L41:
L42:
L43:
L44:
L45:
L46:
L47:
L48:
L49:
L50:
L51:
L52:
L53:
L54:
L55:
L56:
L57:
L58:
L59:
L60:
L61:
L62:
L63:
L64:
L65:
L66:
L67:
L68:
L69:
L70:
L71:
L72:
L73:
L74:
L75:
L76:
L77:
L78:
L79:
L80:
L81:
L82:
L83:
L84:
L85:
L86:
L87:
L88:
L89:
L90:
L91:
L92:
L93:
L94:
L95:
L96:
L97:
L98:
L99:
L100:
	// use the labels to silence compiler errors
	goto L1
	goto L2
	goto L3
	goto L4
	goto L5
	goto L6
	goto L7
	goto L8
	goto L9
	goto L10
	goto L11
	goto L12
	goto L13
	goto L14
	goto L15
	goto L16
	goto L17
	goto L18
	goto L19
	goto L20
	goto L21
	goto L22
	goto L23
	goto L24
	goto L25
	goto L26
	goto L27
	goto L28
	goto L29
	goto L30
	goto L31
	goto L32
	goto L33
	goto L34
	goto L35
	goto L36
	goto L37
	goto L38
	goto L39
	goto L40
	goto L41
	goto L42
	goto L43
	goto L44
	goto L45
	goto L46
	goto L47
	goto L48
	goto L49
	goto L50
	goto L51
	goto L52
	goto L53
	goto L54
	goto L55
	goto L56
	goto L57
	goto L58
	goto L59
	goto L60
	goto L61
	goto L62
	goto L63
	goto L64
	goto L65
	goto L66
	goto L67
	goto L68
	goto L69
	goto L70
	goto L71
	goto L72
	goto L73
	goto L74
	goto L75
	goto L76
	goto L77
	goto L78
	goto L79
	goto L80
	goto L81
	goto L82
	goto L83
	goto L84
	goto L85
	goto L86
	goto L87
	goto L88
	goto L89
	goto L90
	goto L91
	goto L92
	goto L93
	goto L94
	goto L95
	goto L96
	goto L97
	goto L98
	goto L99
	goto L100
}

func foo121() {
	for i := 0; i < 10; i++ {
		defer myprint(nil, i) // ERROR "... argument escapes to heap$" "i escapes to heap$"
		go myprint(nil, i)    // ERROR "... argument escapes to heap$" "i escapes to heap$"
	}
}

// same as foo121 but check across import
func foo121b() {
	for i := 0; i < 10; i++ {
		defer fmt.Printf("%d", i) // ERROR "... argument escapes to heap$" "i escapes to heap$"
		go fmt.Printf("%d", i)    // ERROR "... argument escapes to heap$" "i escapes to heap$"
	}
}

// a harmless forward jump
func foo122() {
	var i *int

	goto L1
L1:
	i = new(int) // ERROR "foo122 new\(int\) does not escape$"
	_ = i
}

// a backward jump, increases loopdepth
func foo123() {
	var i *int

L1:
	i = new(int) // ERROR "new\(int\) escapes to heap$"

	goto L1
	_ = i
}

func foo124(x **int) { // ERROR "foo124 x does not escape$"
	var i int // ERROR "moved to heap: i$"
	p := &i   // ERROR "&i escapes to heap$"
	func() {  // ERROR "foo124 func literal does not escape$"
		*x = p // ERROR "leaking closure reference p$"
	}()
}

func foo125(ch chan *int) { // ERROR "foo125 ch does not escape$"
	var i int // ERROR "moved to heap: i$"
	p := &i   // ERROR "&i escapes to heap$"
	func() {  // ERROR "foo125 func literal does not escape$"
		ch <- p // ERROR "leaking closure reference p$"
	}()
}

func foo126() {
	var px *int // loopdepth 0
	for {
		// loopdepth 1
		var i int // ERROR "moved to heap: i$"
		func() {  // ERROR "foo126 func literal does not escape$"
			px = &i // ERROR "&i escapes to heap$" "leaking closure reference i"
		}()
	}
	_ = px
}

var px *int

func foo127() {
	var i int // ERROR "moved to heap: i$"
	p := &i   // ERROR "&i escapes to heap$"
	q := p
	px = q
}

func foo128() {
	var i int
	p := &i // ERROR "foo128 &i does not escape$"
	q := p
	_ = q
}

func foo129() {
	var i int // ERROR "moved to heap: i$"
	p := &i   // ERROR "&i escapes to heap$"
	func() {  // ERROR "foo129 func literal does not escape$"
		q := p   // ERROR "leaking closure reference p$"
		func() { // ERROR "foo129.func1 func literal does not escape$"
			r := q // ERROR "leaking closure reference q$"
			px = r
		}()
	}()
}

func foo130() {
	for {
		var i int // ERROR "moved to heap: i$"
		func() {  // ERROR "foo130 func literal does not escape$"
			px = &i // ERROR "&i escapes to heap$" "leaking closure reference i$"
		}()
	}
}

func foo131() {
	var i int // ERROR "moved to heap: i$"
	func() {  // ERROR "foo131 func literal does not escape$"
		px = &i // ERROR "&i escapes to heap$" "leaking closure reference i$"
	}()
}

func foo132() {
	var i int   // ERROR "moved to heap: i$"
	go func() { // ERROR "func literal escapes to heap$"
		px = &i // ERROR "&i escapes to heap$" "leaking closure reference i$"
	}()
}

func foo133() {
	var i int      // ERROR "moved to heap: i$"
	defer func() { // ERROR "foo133 func literal does not escape$"
		px = &i // ERROR "&i escapes to heap$" "leaking closure reference i$"
	}()
}

func foo134() {
	var i int
	p := &i  // ERROR "foo134 &i does not escape$"
	func() { // ERROR "foo134 func literal does not escape$"
		q := p
		func() { // ERROR "foo134.func1 func literal does not escape$"
			r := q
			_ = r
		}()
	}()
}

func foo135() {
	var i int   // ERROR "moved to heap: i$"
	p := &i     // ERROR "&i escapes to heap$"
	go func() { // ERROR "func literal escapes to heap$"
		q := p
		func() { // ERROR "foo135.func1 func literal does not escape$"
			r := q
			_ = r
		}()
	}()
}

func foo136() {
	var i int   // ERROR "moved to heap: i$"
	p := &i     // ERROR "&i escapes to heap$"
	go func() { // ERROR "func literal escapes to heap$"
		q := p   // ERROR "leaking closure reference p$"
		func() { // ERROR "foo136.func1 func literal does not escape$"
			r := q // ERROR "leaking closure reference q$"
			px = r
		}()
	}()
}

func foo137() {
	var i int // ERROR "moved to heap: i$"
	p := &i   // ERROR "&i escapes to heap$"
	func() {  // ERROR "foo137 func literal does not escape$"
		q := p      // ERROR "leaking closure reference p$"
		go func() { // ERROR "func literal escapes to heap$"
			r := q
			_ = r
		}()
	}()
}

func foo138() *byte {
	type T struct {
		x [1]byte
	}
	t := new(T)    // ERROR "new\(T\) escapes to heap$"
	return &t.x[0] // ERROR "&t.x\[0\] escapes to heap$"
}

func foo139() *byte {
	type T struct {
		x struct {
			y byte
		}
	}
	t := new(T)   // ERROR "new\(T\) escapes to heap$"
	return &t.x.y // ERROR "&t.x.y escapes to heap$"
}

// issue 4751
func foo140() interface{} {
	type T struct {
		X string
	}
	type U struct {
		X string
		T *T
	}
	t := &T{} // ERROR "&T literal escapes to heap$"
	return U{ // ERROR "U literal escapes to heap$"
		X: t.X,
		T: t,
	}
}

//go:noescape

func F1([]byte)

func F2([]byte)

//go:noescape

func F3(x []byte) // ERROR "F3 x does not escape$"

func F4(x []byte)

func G() {
	var buf1 [10]byte
	F1(buf1[:]) // ERROR "G buf1 does not escape$"

	var buf2 [10]byte // ERROR "moved to heap: buf2$"
	F2(buf2[:])       // ERROR "buf2 escapes to heap$"

	var buf3 [10]byte
	F3(buf3[:]) // ERROR "G buf3 does not escape$"

	var buf4 [10]byte // ERROR "moved to heap: buf4$"
	F4(buf4[:])       // ERROR "buf4 escapes to heap$"
}

type Tm struct {
	x int
}

func (t *Tm) M() { // ERROR "\(\*Tm\).M t does not escape$"
}

func foo141() {
	var f func()

	t := new(Tm) // ERROR "new\(Tm\) escapes to heap$"
	f = t.M      // ERROR "foo141 t.M does not escape$"
	_ = f
}

var gf func()

func foo142() {
	t := new(Tm) // ERROR "new\(Tm\) escapes to heap$"
	gf = t.M     // ERROR "t.M escapes to heap$"
}

// issue 3888.
func foo143() {
	for i := 0; i < 1000; i++ {
		func() { // ERROR "foo143 func literal does not escape$"
			for i := 0; i < 1; i++ {
				var t Tm
				t.M() // ERROR "foo143.func1 t does not escape$"
			}
		}()
	}
}

// issue 5773
// Check that annotations take effect regardless of whether they
// are before or after the use in the source code.

//go:noescape

func foo144a(*int)

func foo144() {
	var x int
	foo144a(&x) // ERROR "foo144 &x does not escape$"
	var y int
	foo144b(&y) // ERROR "foo144 &y does not escape$"
}

//go:noescape

func foo144b(*int)

// issue 7313: for loop init should not be treated as "in loop"

type List struct {
	Next *List
}

func foo145(l List) { // ERROR "foo145 l does not escape$"
	var p *List
	for p = &l; p.Next != nil; p = p.Next { // ERROR "foo145 &l does not escape$"
	}
}

func foo146(l List) { // ERROR "foo146 l does not escape$"
	var p *List
	p = &l // ERROR "foo146 &l does not escape$"
	for ; p.Next != nil; p = p.Next {
	}
}

func foo147(l List) { // ERROR "foo147 l does not escape$"
	var p *List
	p = &l // ERROR "foo147 &l does not escape$"
	for p.Next != nil {
		p = p.Next
	}
}

func foo148(l List) { // ERROR "foo148 l does not escape$"
	for p := &l; p.Next != nil; p = p.Next { // ERROR "foo148 &l does not escape$"
	}
}

// related: address of variable should have depth of variable, not of loop

func foo149(l List) { // ERROR "foo149 l does not escape$"
	var p *List
	for {
		for p = &l; p.Next != nil; p = p.Next { // ERROR "foo149 &l does not escape$"
		}
	}
}

// issue 7934: missed ... if element type had no pointers

var save150 []byte

func foo150(x ...byte) { // ERROR "leaking param: x$"
	save150 = x
}

func bar150() {
	foo150(1, 2, 3) // ERROR "... argument escapes to heap$"
}

// issue 7931: bad handling of slice of array

var save151 *int

func foo151(x *int) { // ERROR "leaking param: x$"
	save151 = x
}

func bar151() {
	var a [64]int // ERROR "moved to heap: a$"
	a[4] = 101
	foo151(&(&a)[4:8][0]) // ERROR "&\(&a\)\[4:8\]\[0\] escapes to heap$" "&a escapes to heap$"
}

func bar151b() {
	var a [10]int      // ERROR "moved to heap: a$"
	b := a[:]          // ERROR "a escapes to heap$"
	foo151(&b[4:8][0]) // ERROR "&b\[4:8\]\[0\] escapes to heap$"
}

func bar151c() {
	var a [64]int // ERROR "moved to heap: a$"
	a[4] = 101
	foo151(&(&a)[4:8:8][0]) // ERROR "&\(&a\)\[4:8:8\]\[0\] escapes to heap$" "&a escapes to heap$"
}

func bar151d() {
	var a [10]int        // ERROR "moved to heap: a$"
	b := a[:]            // ERROR "a escapes to heap$"
	foo151(&b[4:8:8][0]) // ERROR "&b\[4:8:8\]\[0\] escapes to heap$"
}

// issue 8120

type U struct {
	s *string
}

func (u *U) String() *string { // ERROR "leaking param: u to result ~r0 level=1$"
	return u.s
}

type V struct {
	s *string
}

// BAD -- level of leak ought to be 0
func NewV(u U) *V { // ERROR "leaking param: u to result ~r1 level=-1"
	return &V{u.String()} // ERROR "&V literal escapes to heap$" "NewV u does not escape"
}

func foo152() {
	a := "a"   // ERROR "moved to heap: a$"
	u := U{&a} // ERROR "&a escapes to heap$"
	v := NewV(u)
	println(v)
}

// issue 8176 - &x in type switch body not marked as escaping

func foo153(v interface{}) *int { // ERROR "leaking param: v to result ~r1 level=-1$"
	switch x := v.(type) {
	case int: // ERROR "moved to heap: x$"
		return &x // ERROR "&x escapes to heap$"
	}
	panic(0)
}

// issue 8185 - &result escaping into result

func f() (x int, y *int) { // ERROR "moved to heap: x$"
	y = &x // ERROR "&x escapes to heap$"
	return
}

func g() (x interface{}) { // ERROR "moved to heap: x$"
	x = &x // ERROR "&x escapes to heap$"
	return
}

var sink interface{}

type Lit struct {
	p *int
}

func ptrlitNoescape() {
	// Both literal and element do not escape.
	i := 0
	x := &Lit{&i} // ERROR "ptrlitNoescape &Lit literal does not escape$" "ptrlitNoescape &i does not escape$"
	_ = x
}

func ptrlitNoEscape2() {
	// Literal does not escape, but element does.
	i := 0        // ERROR "moved to heap: i$"
	x := &Lit{&i} // ERROR "&i escapes to heap$" "ptrlitNoEscape2 &Lit literal does not escape$"
	sink = *x     // ERROR "\*x escapes to heap$"
}

func ptrlitEscape() {
	// Both literal and element escape.
	i := 0        // ERROR "moved to heap: i$"
	x := &Lit{&i} // ERROR "&Lit literal escapes to heap$" "&i escapes to heap$"
	sink = x      // ERROR "x escapes to heap$"
}

// self-assignments

type Buffer struct {
	arr  [64]byte
	buf1 []byte
	buf2 []byte
	str1 string
	str2 string
}

func (b *Buffer) foo() { // ERROR "\(\*Buffer\).foo b does not escape$"
	b.buf1 = b.buf1[1:2]   // ERROR "\(\*Buffer\).foo ignoring self-assignment to b.buf1$"
	b.buf1 = b.buf1[1:2:3] // ERROR "\(\*Buffer\).foo ignoring self-assignment to b.buf1$"
	b.buf1 = b.buf2[1:2]   // ERROR "\(\*Buffer\).foo ignoring self-assignment to b.buf1$"
	b.buf1 = b.buf2[1:2:3] // ERROR "\(\*Buffer\).foo ignoring self-assignment to b.buf1$"
}

func (b *Buffer) bar() { // ERROR "leaking param: b$"
	b.buf1 = b.arr[1:2] // ERROR "b.arr escapes to heap$"
}

func (b *Buffer) baz() { // ERROR "\(\*Buffer\).baz b does not escape$"
	b.str1 = b.str1[1:2] // ERROR "\(\*Buffer\).baz ignoring self-assignment to b.str1$"
	b.str1 = b.str2[1:2] // ERROR "\(\*Buffer\).baz ignoring self-assignment to b.str1$"
}

func (b *Buffer) bat() { // ERROR "leaking param content: b$"
	o := new(Buffer) // ERROR "new\(Buffer\) escapes to heap$"
	o.buf1 = b.buf1[1:2]
	sink = o // ERROR "o escapes to heap$"
}

func quux(sp *string, bp *[]byte) { // ERROR "quux bp does not escape$" "quux sp does not escape$"
	*sp = (*sp)[1:2] // ERROR "quux ignoring self-assignment to \*sp$"
	*bp = (*bp)[1:2] // ERROR "quux ignoring self-assignment to \*bp$"
}

type StructWithString struct {
	p *int
	s string
}

// This is escape analysis false negative.
// We assign the pointer to x.p but leak x.s. Escape analysis coarsens flows
// to just x, and thus &i looks escaping.
func fieldFlowTracking() {
	var x StructWithString
	i := 0     // ERROR "moved to heap: i$"
	x.p = &i   // ERROR "&i escapes to heap$"
	sink = x.s // ERROR "x.s escapes to heap$"
}

// String operations.

func slicebytetostring0() {
	b := make([]byte, 20) // ERROR "slicebytetostring0 make\(\[\]byte, 20\) does not escape$"
	s := string(b)        // ERROR "slicebytetostring0 string\(b\) does not escape$"
	_ = s
}

func slicebytetostring1() {
	b := make([]byte, 20) // ERROR "slicebytetostring1 make\(\[\]byte, 20\) does not escape$"
	s := string(b)        // ERROR "slicebytetostring1 string\(b\) does not escape$"
	s1 := s[0:1]
	_ = s1
}

func slicebytetostring2() {
	b := make([]byte, 20) // ERROR "slicebytetostring2 make\(\[\]byte, 20\) does not escape$"
	s := string(b)        // ERROR "string\(b\) escapes to heap$"
	s1 := s[0:1]          // ERROR "moved to heap: s1$"
	sink = &s1            // ERROR "&s1 escapes to heap$"
}

func slicebytetostring3() {
	b := make([]byte, 20) // ERROR "slicebytetostring3 make\(\[\]byte, 20\) does not escape$"
	s := string(b)        // ERROR "string\(b\) escapes to heap$"
	s1 := s[0:1]
	sink = s1 // ERROR "s1 escapes to heap$"
}

func addstr0() {
	s0 := "a"
	s1 := "b"
	s := s0 + s1 // ERROR "addstr0 s0 \+ s1 does not escape$"
	_ = s
}

func addstr1() {
	s0 := "a"
	s1 := "b"
	s := "c"
	s += s0 + s1 // ERROR "addstr1 s0 \+ s1 does not escape$"
	_ = s
}

func addstr2() {
	b := make([]byte, 20) // ERROR "addstr2 make\(\[\]byte, 20\) does not escape$"
	s0 := "a"
	s := string(b) + s0 // ERROR "addstr2 string\(b\) \+ s0 does not escape$" "addstr2 string\(b\) does not escape$"
	_ = s
}

func addstr3() {
	s0 := "a"
	s1 := "b"
	s := s0 + s1 // ERROR "s0 \+ s1 escapes to heap$"
	s2 := s[0:1]
	sink = s2 // ERROR "s2 escapes to heap$"
}

func intstring0() bool {
	// string does not escape
	x := '0'
	s := string(x) // ERROR "intstring0 string\(x\) does not escape$"
	return s == "0"
}

func intstring1() string {
	// string does not escape, but the buffer does
	x := '0'
	s := string(x) // ERROR "string\(x\) escapes to heap$"
	return s
}

func intstring2() {
	// string escapes to heap
	x := '0'
	s := string(x) // ERROR "moved to heap: s$" "string\(x\) escapes to heap$"
	sink = &s      // ERROR "&s escapes to heap$"
}

func stringtoslicebyte0() {
	s := "foo"
	x := []byte(s) // ERROR "stringtoslicebyte0 \(\[\]byte\)\(s\) does not escape$"
	_ = x
}

func stringtoslicebyte1() []byte {
	s := "foo"
	return []byte(s) // ERROR "\(\[\]byte\)\(s\) escapes to heap$"
}

func stringtoslicebyte2() {
	s := "foo"
	sink = []byte(s) // ERROR "\(\[\]byte\)\(s\) escapes to heap$"
}

func stringtoslicerune0() {
	s := "foo"
	x := []rune(s) // ERROR "stringtoslicerune0 \(\[\]rune\)\(s\) does not escape$"
	_ = x
}

func stringtoslicerune1() []rune {
	s := "foo"
	return []rune(s) // ERROR "\(\[\]rune\)\(s\) escapes to heap$"
}

func stringtoslicerune2() {
	s := "foo"
	sink = []rune(s) // ERROR "\(\[\]rune\)\(s\) escapes to heap$"
}

func slicerunetostring0() {
	r := []rune{1, 2, 3} // ERROR "slicerunetostring0 \[\]rune literal does not escape$"
	s := string(r)       // ERROR "slicerunetostring0 string\(r\) does not escape$"
	_ = s
}

func slicerunetostring1() string {
	r := []rune{1, 2, 3} // ERROR "slicerunetostring1 \[\]rune literal does not escape$"
	return string(r)     // ERROR "string\(r\) escapes to heap$"
}

func slicerunetostring2() {
	r := []rune{1, 2, 3} // ERROR "slicerunetostring2 \[\]rune literal does not escape$"
	sink = string(r)     // ERROR "string\(r\) escapes to heap$"
}

func makemap0() {
	m := make(map[int]int) // ERROR "makemap0 make\(map\[int\]int\) does not escape$"
	m[0] = 0
	m[1]++
	delete(m, 1)
	sink = m[0] // ERROR "m\[0\] escapes to heap$"
}

func makemap1() map[int]int {
	return make(map[int]int) // ERROR "make\(map\[int\]int\) escapes to heap$"
}

func makemap2() {
	m := make(map[int]int) // ERROR "make\(map\[int\]int\) escapes to heap$"
	sink = m               // ERROR "m escapes to heap$"
}

func nonescapingEface(m map[interface{}]bool) bool { // ERROR "nonescapingEface m does not escape$"
	return m["foo"] // ERROR "nonescapingEface .foo. does not escape$"
}

func nonescapingIface(m map[M]bool) bool { // ERROR "nonescapingIface m does not escape$"
	return m[MV(0)] // ERROR "nonescapingIface MV\(0\) does not escape$"
}

func issue10353() {
	x := new(int) // ERROR "new\(int\) escapes to heap$"
	issue10353a(x)()
}

func issue10353a(x *int) func() { // ERROR "leaking param: x to result ~r1 level=-1$"
	return func() { // ERROR "func literal escapes to heap$"
		println(*x)
	}
}

func issue10353b() {
	var f func()
	for {
		x := new(int) // ERROR "new\(int\) escapes to heap$"
		f = func() {  // ERROR "func literal escapes to heap$"
			println(*x)
		}
	}
	_ = f
}

func issue11387(x int) func() int {
	f := func() int { return x }    // ERROR "func literal escapes to heap"
	slice1 := []func() int{f}       // ERROR "\[\].* does not escape"
	slice2 := make([]func() int, 1) // ERROR "make\(.*\) does not escape"
	copy(slice2, slice1)
	return slice2[0]
}
