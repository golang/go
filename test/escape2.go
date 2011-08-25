// errchk -0 $G -sm $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "unsafe"

var gxx *int

func foo1(x int) {  // ERROR "moved to heap: NAME-x"
	gxx = &x
}

func foo2(yy *int) {  // ERROR "leaking param: NAME-yy"
	gxx = yy
}

func foo3(x int) *int {  // ERROR "moved to heap: NAME-x"
	return &x
}

type T *T
func foo3b(t T) {  // ERROR "leaking param: NAME-t"
	*t = t
}

// xx isn't going anywhere, so use of yy is ok
func foo4(xx, yy *int) {
	xx = yy
}

// xx isn't going anywhere, so taking address of yy is ok
func foo5(xx **int, yy *int) {
	xx = &yy
}

func foo6(xx **int, yy *int) {  // ERROR "leaking param: NAME-yy"
	*xx = yy
}

func foo7(xx **int, yy *int) {
	**xx = *yy
}

func foo8(xx, yy *int) int {
	xx = yy
	return *xx
}

func foo9(xx, yy *int) *int {  // ERROR "leaking param: NAME-xx" "leaking param: NAME-yy"
	xx = yy
	return xx
}

func foo10(xx, yy *int) {
	*xx = *yy
}

func foo11() int {
	x, y := 0, 42
	xx := &x
	yy := &y
	*xx = *yy
	return x
}


var xxx **int

func foo12(yyy **int) {   // ERROR "leaking param: NAME-yyy"
	xxx = yyy
}

func foo13(yyy **int) {
	*xxx = *yyy
}

func foo14(yyy **int) {
	**xxx = **yyy
}

func foo15(yy *int) {  // ERROR "moved to heap: NAME-yy"
	xxx = &yy
}

func foo16(yy *int) {  // ERROR "leaking param: NAME-yy"
	*xxx = yy
}

func foo17(yy *int) {
	**xxx = *yy
}

func foo18(y int) {  // ERROR "moved to heap: "NAME-y"
	*xxx = &y
}

func foo19(y int) {
	**xxx = y
}

type Bar struct {
	i int
	ii *int
}

func NewBar() *Bar {
	return &Bar{ 42, nil }
}

func NewBarp(x *int) *Bar {  // ERROR "leaking param: NAME-x"
	return &Bar{ 42, x }
}

func NewBarp2(x *int) *Bar {
	return &Bar{ *x, nil }
}

func (b *Bar) NoLeak() int {
	return *(b.ii)
}

func (b *Bar) AlsoNoLeak() *int {
	return b.ii
}

type Bar2 struct {
	i [12]int
	ii []int
}

func NewBar2() *Bar2 {
	return &Bar2{ [12]int{ 42 },  nil }
}

func (b *Bar2) NoLeak() int {
	return b.i[0]
}

func (b *Bar2) Leak() []int {  // ERROR "leaking param: NAME-b"
	return b.i[:]
}

func (b *Bar2) AlsoNoLeak() []int {
	return b.ii[0:1]
}

func (b *Bar2) LeakSelf() {  // ERROR "leaking param: NAME-b"
	b.ii = b.i[0:4]
}

func (b *Bar2) LeakSelf2() {  // ERROR "leaking param: NAME-b"
	var buf []int
	buf = b.i[0:]
	b.ii = buf
}

func foo21() func() int {
	x := 42  // ERROR "moved to heap: NAME-x"
	return func() int {
		return x
	}
}

func foo22() int {
	x := 42
	return func() int {
		return x
	}()
}

func foo23(x int) func() int {  // ERROR "moved to heap: NAME-x"
	return func() int {
		return x
	}
}

func foo23a(x int) (func() int) {  // ERROR "moved to heap: NAME-x"
	f := func() int {
		return x
	}
	return f
}

func foo23b(x int) *(func() int) {  // ERROR "moved to heap: NAME-x"
	f := func() int { return x } // ERROR "moved to heap: NAME-f"
	return &f
}

func foo24(x int) int {
	return func() int {
		return x
	}()
}


var x *int

func fooleak(xx *int) int {    // ERROR "leaking param: NAME-xx"
	x = xx
	return *x
}

func foonoleak(xx *int) int {
	return *x + *xx
}

func foo31(x int) int {  // ERROR "moved to heap: NAME-x"
	return fooleak(&x)
}

func foo32(x int) int {
	return foonoleak(&x)
}

type Foo struct {
	xx *int
	x int
}

var F Foo
var pf *Foo

func (f *Foo) fooleak() {  // ERROR "leaking param: NAME-f"
	pf = f
}

func (f *Foo) foonoleak() {
	F.x = f.x
}

func (f *Foo) Leak() {  // ERROR "leaking param: NAME-f"
	f.fooleak()
}

func (f *Foo) NoLeak() {
	f.foonoleak()
}


func foo41(x int) {  // ERROR "moved to heap: NAME-x"
	F.xx = &x
}

func (f *Foo) foo42(x int) {   // ERROR "moved to heap: NAME-x"
	f.xx = &x
}

func foo43(f *Foo, x int) {   // ERROR "moved to heap: NAME-x"
	f.xx = &x
}

func foo44(yy *int) {  // ERROR "leaking param: NAME-yy"
	F.xx = yy
}

func (f *Foo) foo45() {
	F.x = f.x 
}

func (f *Foo) foo46() {
	F.xx = f.xx 
}

func (f *Foo) foo47() {  // ERROR "leaking param: NAME-f"
	f.xx = &f.x
}


var ptrSlice []*int

func foo50(i *int) {  // ERROR "leaking param: NAME-i"
	ptrSlice[0] = i
}


var ptrMap map[*int]*int

func foo51(i *int) {   // ERROR "leaking param: NAME-i"
	ptrMap[i] = i
}


func indaddr1(x int) *int { // ERROR "moved to heap: NAME-x"
	return &x
}

func indaddr2(x *int) *int {   // ERROR "leaking param: NAME-x"
	return *&x
}

func indaddr3(x *int32) *int {    // ERROR "leaking param: NAME-x"
	return *(**int)(unsafe.Pointer(&x))
}

// From package math:

func Float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}

func Float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}

func Float64bits(f float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&f))
}

func Float64frombits(b uint64) float64 {
	return *(*float64)(unsafe.Pointer(&b))
}

// contrast with
func float64bitsptr(f float64) *uint64 {  // ERROR "moved to heap: NAME-f"
	return (*uint64)(unsafe.Pointer(&f))
}

func float64ptrbitsptr(f *float64) *uint64 {  // ERROR "leaking param: NAME-f"
	return (*uint64)(unsafe.Pointer(f))
}

func typesw(i interface{}) *int {  // ERROR "leaking param: NAME-i"
	switch val := i.(type) {
	case *int:
		return val
	case *int8:
		v := int(*val)  // ERROR "moved to heap: NAME-v"
		return &v
	}
	return nil
}

func exprsw(i *int) *int {	// ERROR "leaking param: NAME-i"
	switch j := i; *j + 110 {
	case 12:
		return j
	case 42:
		return nil
	}
	return nil

}

// assigning to an array element is like assigning to the array
func foo60(i *int) *int {  // ERROR "leaking param: NAME-i"
	var a [12]*int
	a[0] = i
	return a[1]
}

func foo60a(i *int) *int {
	var a [12]*int
	a[0] = i
	return nil
}

// assigning to a struct field  is like assigning to the struct
func foo61(i *int) *int {   // ERROR "leaking param: NAME-i"
	type S struct {
		a,b *int
	}
	var s S
	s.a = i
	return s.b
}

func foo61a(i *int) *int {
	type S struct {
		a,b *int
	}
	var s S
	s.a = i
	return nil
}

// assigning to a struct field is like assigning to the struct but
// here this subtlety is lost, since s.a counts as an assignment to a
// track-losing dereference.
func foo62(i *int) *int {   // ERROR "leaking param: NAME-i"
	type S struct {
		a,b *int
	}
	s := new(S)
	s.a = i
	return nil  // s.b
}


type M interface { M() }

func foo63(m M) {
}

func foo64(m M) {  // ERROR "leaking param: NAME-m"
	m.M()
}

type MV int
func (MV) M() {}

func foo65() {
	var mv MV
	foo63(&mv)
}

func foo66() {
	var mv MV  // ERROR "moved to heap: NAME-mv"
	foo64(&mv)
}

func foo67() {
	var mv MV
	foo63(mv)
}

func foo68() {
	var mv MV
	foo64(mv)  // escapes but it's an int so irrelevant
}

func foo69(m M) {  // ERROR "leaking param: NAME-m"
	foo64(m)
}

func foo70(mv1 *MV, m M) {  // ERROR "leaking param: NAME-mv1" "leaking param: NAME-m"
	m = mv1
	foo64(m)
}

func foo71(x *int) []*int {  // ERROR "leaking param: NAME-x"
	var y []*int
	y = append(y, x)
	return y
}

func foo71a(x int) []*int {  // ERROR "moved to heap: NAME-x"
	var y []*int
	y = append(y, &x)
	return y
}

func foo72() {
	var x int
	var y [1]*int
	y[0] = &x
}

func foo72aa() [10]*int {
	var x int  // ERROR "moved to heap: NAME-x"
	var y [10]*int
	y[0] = &x
	return y
}

func foo72a() {
	var y [10]*int
	for i := 0; i < 10; i++ {
		x := i  // not moved to heap b/c y goes nowhere
		y[i] = &x
	}
	return
}

func foo72b() [10]*int {
	var y [10]*int
	for i := 0; i < 10; i++ {
		x := i  // ERROR "moved to heap: NAME-x"
		y[i] = &x
	}
	return y
}


// issue 2145
func foo73() {
	s := []int{3,2,1}
	for _, v := range s {
		vv := v  // ERROR "moved to heap: NAME-vv"
		defer func() {  //  "func literal escapes its scope" "&vv escapes its scope"
			println(vv)
		}()
	}
}

func foo74() {
	s := []int{3,2,1}
	for _, v := range s {
		vv := v  // ERROR "moved to heap: NAME-vv"
		fn := func() {  //  "func literal escapes its scope" "&vv escapes its scope"
			println(vv)
		}
		defer fn()
	}
}

func myprint(y *int, x ...interface{}) *int {  // ERROR "leaking param: NAME-y"
	return y
}

func myprint1(y *int, x ...interface{}) *interface{} {  // ERROR "leaking param: NAME-x"
	return &x[0]
}

func foo75(z *int) { // ERROR "leaking param: NAME-z"
	myprint(z, 1, 2, 3)
}

func foo75a(z *int) {
	myprint1(z, 1, 2, 3)  // "[.][.][.] argument escapes to heap"
}

func foo76(z *int) {
	myprint(nil, z)
}

func foo76a(z *int) {  // ERROR "leaking param: NAME-z"
	myprint1(nil, z)  // "[.][.][.] argument escapes to heap"
}

func foo76b() {
	myprint(nil, 1, 2, 3)
}

func foo76c() {
	myprint1(nil, 1, 2, 3) // "[.][.][.] argument escapes to heap"
}

func foo76d() {
	defer myprint(nil, 1, 2, 3)
}

func foo76e() {
	defer myprint1(nil, 1, 2, 3) // "[.][.][.] argument escapes to heap"
}

func foo76f() {
	for {
		defer myprint(nil, 1, 2, 3) // "[.][.][.] argument escapes its scope"
	}
}

func foo76g() {
	for {
		defer myprint1(nil, 1, 2, 3) // "[.][.][.] argument escapes to heap"
	}
}

func foo77(z []interface{}) {
	myprint(nil, z...)  // z does not escape
}

func foo77a(z []interface{}) {  // ERROR "leaking param: NAME-z"
	myprint1(nil, z...)
}

func foo78(z int) *int {  // ERROR "moved to heap: NAME-z"
	return &z  //  "&z escapes"
}

func foo78a(z int) *int {  // ERROR "moved to heap: NAME-z"
	y := &z
	x := &y
	return *x  // really return y
}

func foo79() *int {
	return new(int)  //  "moved to heap: new[(]int[)]"
}

func foo80() *int {
	var z *int
	for {
		z = new(int) //  "new[(]int[)] escapes its scope"
	}
	_ = z
	return nil
}

func foo81() *int {
	for {
		z := new(int)
		_ = z
	}
	return nil
}

type Fooer interface {
	Foo()
}

type LimitedFooer struct {
        Fooer
        N int64
}

func LimitFooer(r Fooer, n int64) Fooer {  // ERROR "leaking param: NAME-r"
	return &LimitedFooer{r, n}
}

func foo90(x *int) map[*int]*int {  // ERROR "leaking param: NAME-x"
	return map[*int]*int{ nil: x }
}

func foo91(x *int) map[*int]*int {  // ERROR "leaking param: NAME-x"
	return map[*int]*int{ x:nil }
}

func foo92(x *int) [2]*int {  // ERROR "leaking param: NAME-x"
	return [2]*int{ x, nil }
}

// does not leak c
func foo93(c chan *int) *int {
	for v := range c {
		return v
	}
	return nil
}

// does not leak m
func foo94(m map[*int]*int, b bool) *int {
	for k, v := range m {
		if b {
			return k
		}
		return v
	}
	return nil
}

// does leak x
func foo95(m map[*int]*int, x *int) {  // ERROR "leaking param: NAME-x"
	m[x] = x
}

// does not leak m
func foo96(m []*int) *int {
	return m[0]
}

// does leak m
func foo97(m [1]*int) *int {  // ERROR "leaking param: NAME-m"
	return m[0]
}

// does not leak m
func foo98(m map[int]*int) *int {
	return m[0]
}

// does leak m
func foo99(m *[1]*int) []*int {  // ERROR "leaking param: NAME-m"
	return m[:]
}

// does not leak m
func foo100(m []*int) *int {
	for _, v := range m {
		return v
	}
	return nil
}

// does leak m
func foo101(m [1]*int) *int {  // ERROR "leaking param: NAME-m"
	for _, v := range m {
		return v
	}
	return nil
}

// does leak x
func foo102(m []*int, x *int) {  // ERROR "leaking param: NAME-x"
	m[0] = x
}

// does not leak x
func foo103(m [1]*int, x *int) {
	m[0] = x
}

var y []*int

// does not leak x
func foo104(x []*int) {
	copy(y, x)
}

// does not leak x
func foo105(x []*int) {
	_ = append(y, x...)
}

// does leak x
func foo106(x *int) {  // ERROR "leaking param: NAME-x"
	_ = append(y, x)
}
