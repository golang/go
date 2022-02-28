// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// builtin calls

package builtins

import "unsafe"

func f0() {}

func append1() {
	var b byte
	var x int
	var s []byte
	_ = append() // ERROR not enough arguments
	_ = append("foo" /* ERROR must be a slice */ )
	_ = append(nil /* ERROR must be a slice */ , s)
	_ = append(x /* ERROR must be a slice */ , s)
	_ = append(s)
	_ = append(s, nil...)
	append /* ERROR not used */ (s)

	_ = append(s, b)
	_ = append(s, x /* ERROR cannot use x */ )
	_ = append(s, s /* ERROR cannot use s */ )
	_ = append(s /* ERROR not enough arguments */ ...)
	_ = append(s, b, s /* ERROR too many arguments */ ... )
	_ = append(s, 1, 2, 3)
	_ = append(s, 1, 2, 3, x /* ERROR cannot use x */ , 5, 6, 6)
	_ = append(s, 1, 2 /* ERROR too many arguments */ , s... )
	_ = append([]interface{}(nil), 1, 2, "foo", x, 3.1425, false)

	type S []byte
	type T string
	var t T
	_ = append(s, "foo" /* ERROR cannot use .* in argument to append */ )
	_ = append(s, "foo"...)
	_ = append(S(s), "foo" /* ERROR cannot use .* in argument to append */ )
	_ = append(S(s), "foo"...)
	_ = append(s, t /* ERROR cannot use t */ )
	_ = append(s, t...)
	_ = append(s, T("foo")...)
	_ = append(S(s), t /* ERROR cannot use t */ )
	_ = append(S(s), t...)
	_ = append(S(s), T("foo")...)
	_ = append([]string{}, t /* ERROR cannot use t */ , "foo")
	_ = append([]T{}, t, "foo")
}

// from the spec
func append2() {
	s0 := []int{0, 0}
	s1 := append(s0, 2)                // append a single element     s1 == []int{0, 0, 2}
	s2 := append(s1, 3, 5, 7)          // append multiple elements    s2 == []int{0, 0, 2, 3, 5, 7}
	s3 := append(s2, s0...)            // append a slice              s3 == []int{0, 0, 2, 3, 5, 7, 0, 0}
	s4 := append(s3[3:6], s3[2:]...)   // append overlapping slice    s4 == []int{3, 5, 7, 2, 3, 5, 7, 0, 0}

	var t []interface{}
	t = append(t, 42, 3.1415, "foo")   //                             t == []interface{}{42, 3.1415, "foo"}

	var b []byte
	b = append(b, "bar"...)            // append string contents      b == []byte{'b', 'a', 'r' }

	_ = s4
}

func append3() {
	f1 := func() (s []int) { return }
	f2 := func() (s []int, x int) { return }
	f3 := func() (s []int, x, y int) { return }
	f5 := func() (s []interface{}, x int, y float32, z string, b bool) { return }
	ff := func() (int, float32) { return 0, 0 }
	_ = append(f0 /* ERROR used as value */ ())
	_ = append(f1())
	_ = append(f2())
	_ = append(f3())
	_ = append(f5())
	_ = append(ff /* ERROR must be a slice */ ()) // TODO(gri) better error message
}

func cap1() {
	var a [10]bool
	var p *[20]int
	var c chan string
	_ = cap() // ERROR not enough arguments
	_ = cap(1, 2) // ERROR too many arguments
	_ = cap(42 /* ERROR invalid */)
	const _3 = cap(a)
	assert(_3 == 10)
	const _4 = cap(p)
	assert(_4 == 20)
	_ = cap(c)
	cap /* ERROR not used */ (c)

	// issue 4744
	type T struct{ a [10]int }
	const _ = cap(((*T)(nil)).a)

	var s [][]byte
	_ = cap(s)
	_ = cap(s... /* ERROR invalid use of \.\.\. */ )
}

func cap2() {
	f1a := func() (a [10]int) { return }
	f1s := func() (s []int) { return }
	f2 := func() (s []int, x int) { return }
	_ = cap(f0 /* ERROR used as value */ ())
	_ = cap(f1a())
	_ = cap(f1s())
	_ = cap(f2()) // ERROR too many arguments
}

// test cases for issue 7387
func cap3() {
	var f = func() int { return 0 }
	var x = f()
	const (
		_ = cap([4]int{})
		_ = cap([4]int{x})
		_ = cap /* ERROR not constant */ ([4]int{f()})
		_ = cap /* ERROR not constant */ ([4]int{cap([]int{})})
		_ = cap([4]int{cap([4]int{})})
	)
	var y float64
	var z complex128
	const (
		_ = cap([4]float64{})
		_ = cap([4]float64{y})
		_ = cap([4]float64{real(2i)})
		_ = cap /* ERROR not constant */ ([4]float64{real(z)})
	)
	var ch chan [10]int
	const (
		_ = cap /* ERROR not constant */ (<-ch)
		_ = cap /* ERROR not constant */ ([4]int{(<-ch)[0]})
	)
}

func close1() {
	var c chan int
	var r <-chan int
	close() // ERROR not enough arguments
	close(1, 2) // ERROR too many arguments
	close(42 /* ERROR cannot close non-channel */)
	close(r /* ERROR receive-only channel */)
	close(c)
	_ = close /* ERROR used as value */ (c)

	var s []chan int
	close(s... /* ERROR invalid use of \.\.\. */ )
}

func close2() {
	f1 := func() (ch chan int) { return }
	f2 := func() (ch chan int, x int) { return }
	close(f0 /* ERROR used as value */ ())
	close(f1())
	close(f2()) // ERROR too many arguments
}

func complex1() {
	var i32 int32
	var f32 float32
	var f64 float64
	var c64 complex64
	var c128 complex128
	_ = complex() // ERROR not enough arguments
	_ = complex(1) // ERROR not enough arguments
	_ = complex(true /* ERROR mismatched types */ , 0)
	_ = complex(i32 /* ERROR expected floating-point */ , 0)
	_ = complex("foo" /* ERROR mismatched types */ , 0)
	_ = complex(c64 /* ERROR expected floating-point */ , 0)
	_ = complex(0 /* ERROR mismatched types */ , true)
	_ = complex(0 /* ERROR expected floating-point */ , i32)
	_ = complex(0 /* ERROR mismatched types */ , "foo")
	_ = complex(0 /* ERROR expected floating-point */ , c64)
	_ = complex(f32, f32)
	_ = complex(f32, 1)
	_ = complex(f32, 1.0)
	_ = complex(f32, 'a')
	_ = complex(f64, f64)
	_ = complex(f64, 1)
	_ = complex(f64, 1.0)
	_ = complex(f64, 'a')
	_ = complex(f32 /* ERROR mismatched types */ , f64)
	_ = complex(f64 /* ERROR mismatched types */ , f32)
	_ = complex(1, 1)
	_ = complex(1, 1.1)
	_ = complex(1, 'a')
	complex /* ERROR not used */ (1, 2)

	var _ complex64 = complex(f32, f32)
	var _ complex64 = complex /* ERROR cannot use .* in variable declaration */ (f64, f64)

	var _ complex128 = complex /* ERROR cannot use .* in variable declaration */ (f32, f32)
	var _ complex128 = complex(f64, f64)

	// untyped constants
	const _ int = complex(1, 0)
	const _ float32 = complex(1, 0)
	const _ complex64 = complex(1, 0)
	const _ complex128 = complex(1, 0)
	const _ = complex(0i, 0i)
	const _ = complex(0i, 0)
	const _ int = 1.0 + complex(1, 0i)

	const _ int = complex /* ERROR int */ (1.1, 0)
	const _ float32 = complex /* ERROR float32 */ (1, 2)

	// untyped values
	var s uint
	_ = complex(1 /* ERROR integer */ <<s, 0)
	const _ = complex /* ERROR not constant */ (1 /* ERROR integer */ <<s, 0)
	var _ int = complex /* ERROR cannot use .* in variable declaration */ (1 /* ERROR integer */ <<s, 0)

	// floating-point argument types must be identical
	type F32 float32
	type F64 float64
	var x32 F32
	var x64 F64
	c64 = complex(x32, x32)
	_ = complex(x32 /* ERROR mismatched types */ , f32)
	_ = complex(f32 /* ERROR mismatched types */ , x32)
	c128 = complex(x64, x64)
	_ = c128
	_ = complex(x64 /* ERROR mismatched types */ , f64)
	_ = complex(f64 /* ERROR mismatched types */ , x64)

	var t []float32
	_ = complex(t... /* ERROR invalid use of \.\.\. */ )
}

func complex2() {
	f1 := func() (x float32) { return }
	f2 := func() (x, y float32) { return }
	f3 := func() (x, y, z float32) { return }
	_ = complex(f0 /* ERROR used as value */ ())
	_ = complex(f1()) // ERROR not enough arguments
	_ = complex(f2())
	_ = complex(f3()) // ERROR too many arguments
}

func copy1() {
	copy() // ERROR not enough arguments
	copy("foo") // ERROR not enough arguments
	copy([ /* ERROR copy expects slice arguments */ ...]int{}, []int{})
	copy([ /* ERROR copy expects slice arguments */ ]int{}, [...]int{})
	copy([ /* ERROR different element types */ ]int8{}, "foo")

	// spec examples
	var a = [...]int{0, 1, 2, 3, 4, 5, 6, 7}
	var s = make([]int, 6)
	var b = make([]byte, 5)
	n1 := copy(s, a[0:])            // n1 == 6, s == []int{0, 1, 2, 3, 4, 5}
	n2 := copy(s, s[2:])            // n2 == 4, s == []int{2, 3, 4, 5, 4, 5}
	n3 := copy(b, "Hello, World!")  // n3 == 5, b == []byte("Hello")
	_, _, _ = n1, n2, n3

	var t [][]int
	copy(t, t)
	copy(t /* ERROR copy expects slice arguments */ , nil)
	copy(nil /* ERROR copy expects slice arguments */ , t)
	copy(nil /* ERROR copy expects slice arguments */ , nil)
	copy(t... /* ERROR invalid use of \.\.\. */ )
}

func copy2() {
	f1 := func() (a []int) { return }
	f2 := func() (a, b []int) { return }
	f3 := func() (a, b, c []int) { return }
	copy(f0 /* ERROR used as value */ ())
	copy(f1()) // ERROR not enough arguments
	copy(f2())
	copy(f3()) // ERROR too many arguments
}

func delete1() {
	var m map[string]int
	var s string
	delete() // ERROR not enough arguments
	delete(1) // ERROR not enough arguments
	delete(1, 2, 3) // ERROR too many arguments
	delete(m, 0 /* ERROR cannot use */)
	delete(m, s)
	_ = delete /* ERROR used as value */ (m, s)

	var t []map[string]string
	delete(t... /* ERROR invalid use of \.\.\. */ )
}

func delete2() {
	f1 := func() (m map[string]int) { return }
	f2 := func() (m map[string]int, k string) { return }
	f3 := func() (m map[string]int, k string, x float32) { return }
	delete(f0 /* ERROR used as value */ ())
	delete(f1()) // ERROR not enough arguments
	delete(f2())
	delete(f3()) // ERROR too many arguments
}

func imag1() {
	var f32 float32
	var f64 float64
	var c64 complex64
	var c128 complex128
	_ = imag() // ERROR not enough arguments
	_ = imag(1, 2) // ERROR too many arguments
	_ = imag(10)
	_ = imag(2.7182818)
	_ = imag("foo" /* ERROR expected complex */)
	_ = imag('a')
	const _5 = imag(1 + 2i)
	assert(_5 == 2)
	f32 = _5
	f64 = _5
	const _6 = imag(0i)
	assert(_6 == 0)
	f32 = imag(c64)
	f64 = imag(c128)
	f32 = imag /* ERROR cannot use .* in assignment */ (c128)
	f64 = imag /* ERROR cannot use .* in assignment */ (c64)
	imag /* ERROR not used */ (c64)
	_, _ = f32, f64

	// complex type may not be predeclared
	type C64 complex64
	type C128 complex128
	var x64 C64
	var x128 C128
	f32 = imag(x64)
	f64 = imag(x128)

	var a []complex64
	_ = imag(a... /* ERROR invalid use of \.\.\. */ )

	// if argument is untyped, result is untyped
	const _ byte = imag(1.2 + 3i)
	const _ complex128 = imag(1.2 + 3i)

	// lhs constant shift operands are typed as complex128
	var s uint
	_ = imag(1 /* ERROR must be integer */ << s)
}

func imag2() {
	f1 := func() (x complex128) { return }
	f2 := func() (x, y complex128) { return }
	_ = imag(f0 /* ERROR used as value */ ())
	_ = imag(f1())
	_ = imag(f2()) // ERROR too many arguments
}

func len1() {
	const c = "foobar"
	var a [10]bool
	var p *[20]int
	var m map[string]complex128
	_ = len() // ERROR not enough arguments
	_ = len(1, 2) // ERROR too many arguments
	_ = len(42 /* ERROR invalid */)
	const _3 = len(c)
	assert(_3 == 6)
	const _4 = len(a)
	assert(_4 == 10)
	const _5 = len(p)
	assert(_5 == 20)
	_ = len(m)
	len /* ERROR not used */ (c)

	// esoteric case
	var t string
	var hash map[interface{}][]*[10]int
	const n = len /* ERROR not constant */ (hash[recover()][len(t)])
	assert(n == 10) // ok because n has unknown value and no error is reported
	var ch <-chan int
	const nn = len /* ERROR not constant */ (hash[<-ch][len(t)])

	// issue 4744
	type T struct{ a [10]int }
	const _ = len(((*T)(nil)).a)

	var s [][]byte
	_ = len(s)
	_ = len(s... /* ERROR invalid use of \.\.\. */ )
}

func len2() {
	f1 := func() (x []int) { return }
	f2 := func() (x, y []int) { return }
	_ = len(f0 /* ERROR used as value */ ())
	_ = len(f1())
	_ = len(f2()) // ERROR too many arguments
}

// test cases for issue 7387
func len3() {
	var f = func() int { return 0 }
	var x = f()
	const (
		_ = len([4]int{})
		_ = len([4]int{x})
		_ = len /* ERROR not constant */ ([4]int{f()})
		_ = len /* ERROR not constant */ ([4]int{len([]int{})})
		_ = len([4]int{len([4]int{})})
	)
	var y float64
	var z complex128
	const (
		_ = len([4]float64{})
		_ = len([4]float64{y})
		_ = len([4]float64{real(2i)})
		_ = len /* ERROR not constant */ ([4]float64{real(z)})
	)
	var ch chan [10]int
	const (
		_ = len /* ERROR not constant */ (<-ch)
		_ = len /* ERROR not constant */ ([4]int{(<-ch)[0]})
	)
}

func make1() {
	var n int
	var m float32
	var s uint

	_ = make() // ERROR not enough arguments
	_ = make(1 /* ERROR not a type */)
	_ = make(int /* ERROR cannot make */)

	// slices
	_ = make/* ERROR arguments */ ([]int)
	_ = make/* ERROR arguments */ ([]int, 2, 3, 4)
	_ = make([]int, int /* ERROR not an expression */)
	_ = make([]int, 10, float32 /* ERROR not an expression */)
	_ = make([]int, "foo" /* ERROR cannot convert */)
	_ = make([]int, 10, 2.3 /* ERROR truncated */)
	_ = make([]int, 5, 10.0)
	_ = make([]int, 0i)
	_ = make([]int, 1.0)
	_ = make([]int, 1.0<<s)
	_ = make([]int, 1.1 /* ERROR int */ <<s)
	_ = make([]int, - /* ERROR must not be negative */ 1, 10)
	_ = make([]int, 0, - /* ERROR must not be negative */ 1)
	_ = make([]int, - /* ERROR must not be negative */ 1, - /* ERROR must not be negative */ 1)
	_ = make([]int, 1 /* ERROR overflows */ <<100, 1 /* ERROR overflows */ <<100)
	_ = make([]int, 10 /* ERROR length and capacity swapped */ , 9)
	_ = make([]int, 1 /* ERROR overflows */ <<100, 12345)
	_ = make([]int, m /* ERROR must be integer */ )
        _ = &make /* ERROR cannot take address */ ([]int, 0)

	// maps
	_ = make /* ERROR arguments */ (map[int]string, 10, 20)
	_ = make(map[int]float32, int /* ERROR not an expression */)
	_ = make(map[int]float32, "foo" /* ERROR cannot convert */)
	_ = make(map[int]float32, 10)
	_ = make(map[int]float32, n)
	_ = make(map[int]float32, int64(n))
	_ = make(map[string]bool, 10.0)
	_ = make(map[string]bool, 10.0<<s)
        _ = &make /* ERROR cannot take address */ (map[string]bool)

	// channels
	_ = make /* ERROR arguments */ (chan int, 10, 20)
	_ = make(chan int, int /* ERROR not an expression */)
	_ = make(chan<- int, "foo" /* ERROR cannot convert */)
	_ = make(chan int, - /* ERROR must not be negative */ 10)
	_ = make(<-chan float64, 10)
	_ = make(chan chan int, n)
	_ = make(chan string, int64(n))
	_ = make(chan bool, 10.0)
	_ = make(chan bool, 10.0<<s)
        _ = &make /* ERROR cannot take address */ (chan bool)

	make /* ERROR not used */ ([]int, 10)

	var t []int
	_ = make([]int, t[0], t[1])
	_ = make([]int, t... /* ERROR invalid use of \.\.\. */ )
}

func make2() {
	f1 := func() (x []int) { return }
	_ = make(f0 /* ERROR not a type */ ())
	_ = make(f1 /* ERROR not a type */ ())
}

func new1() {
	_ = new() // ERROR not enough arguments
	_ = new(1, 2) // ERROR too many arguments
	_ = new("foo" /* ERROR not a type */)
	p := new(float64)
	_ = new(struct{ x, y int })
	q := new(*float64)
	_ = *p == **q
	new /* ERROR not used */ (int)
        _ = &new /* ERROR cannot take address */ (int)

	_ = new(int... /* ERROR invalid use of \.\.\. */ )
}

func new2() {
	f1 := func() (x []int) { return }
	_ = new(f0 /* ERROR not a type */ ())
	_ = new(f1 /* ERROR not a type */ ())
}

func panic1() {
	panic() // ERROR not enough arguments
	panic(1, 2) // ERROR too many arguments
	panic(0)
	panic("foo")
	panic(false)
	panic(1<<10)
	panic(1 << /* ERROR constant shift overflow */ 1000)
	_ = panic /* ERROR used as value */ (0)

	var s []byte
	panic(s)
	panic(s... /* ERROR invalid use of \.\.\. */ )
}

func panic2() {
	f1 := func() (x int) { return }
	f2 := func() (x, y int) { return }
	panic(f0 /* ERROR used as value */ ())
	panic(f1())
	panic(f2()) // ERROR too many arguments
}

func print1() {
	print()
	print(1)
	print(1, 2)
	print("foo")
	print(2.718281828)
	print(false)
	print(1<<10)
	print(1 << /* ERROR constant shift overflow */ 1000)
	println(nil /* ERROR untyped nil */ )

	var s []int
	print(s... /* ERROR invalid use of \.\.\. */ )
	_ = print /* ERROR used as value */ ()
}

func print2() {
	f1 := func() (x int) { return }
	f2 := func() (x, y int) { return }
	f3 := func() (x int, y float32, z string) { return }
	print(f0 /* ERROR used as value */ ())
	print(f1())
	print(f2())
	print(f3())
}

func println1() {
	println()
	println(1)
	println(1, 2)
	println("foo")
	println(2.718281828)
	println(false)
	println(1<<10)
	println(1 << /* ERROR constant shift overflow */ 1000)
	println(nil /* ERROR untyped nil */ )

	var s []int
	println(s... /* ERROR invalid use of \.\.\. */ )
	_ = println /* ERROR used as value */ ()
}

func println2() {
	f1 := func() (x int) { return }
	f2 := func() (x, y int) { return }
	f3 := func() (x int, y float32, z string) { return }
	println(f0 /* ERROR used as value */ ())
	println(f1())
	println(f2())
	println(f3())
}

func real1() {
	var f32 float32
	var f64 float64
	var c64 complex64
	var c128 complex128
	_ = real() // ERROR not enough arguments
	_ = real(1, 2) // ERROR too many arguments
	_ = real(10)
	_ = real(2.7182818)
	_ = real("foo" /* ERROR expected complex */)
	const _5 = real(1 + 2i)
	assert(_5 == 1)
	f32 = _5
	f64 = _5
	const _6 = real(0i)
	assert(_6 == 0)
	f32 = real(c64)
	f64 = real(c128)
	f32 = real /* ERROR cannot use .* in assignment */ (c128)
	f64 = real /* ERROR cannot use .* in assignment */ (c64)
	real /* ERROR not used */ (c64)

	// complex type may not be predeclared
	type C64 complex64
	type C128 complex128
	var x64 C64
	var x128 C128
	f32 = imag(x64)
	f64 = imag(x128)
	_, _ = f32, f64

	var a []complex64
	_ = real(a... /* ERROR invalid use of \.\.\. */ )

	// if argument is untyped, result is untyped
	const _ byte = real(1 + 2.3i)
	const _ complex128 = real(1 + 2.3i)

	// lhs constant shift operands are typed as complex128
	var s uint
	_ = real(1 /* ERROR must be integer */ << s)
}

func real2() {
	f1 := func() (x complex128) { return }
	f2 := func() (x, y complex128) { return }
	_ = real(f0 /* ERROR used as value */ ())
	_ = real(f1())
	_ = real(f2()) // ERROR too many arguments
}

func recover1() {
	_ = recover()
	_ = recover(10) // ERROR too many arguments
	recover()

	var s []int
	recover(s... /* ERROR invalid use of \.\.\. */ )
}

func recover2() {
	f1 := func() (x int) { return }
	f2 := func() (x, y int) { return }
	_ = recover(f0 /* ERROR used as value */ ())
	_ = recover(f1()) // ERROR too many arguments
	_ = recover(f2()) // ERROR too many arguments
}

// assuming types.DefaultPtrSize == 8
type S0 struct{      // offset
	a bool       //  0
	b rune       //  4
	c *int       //  8
	d bool       // 16
	e complex128 // 24
}                    // 40

type S1 struct{   // offset
	x float32 //  0
	y string  //  8
	z *S1     // 24
	S0        // 32
}                 // 72

type S2 struct{ // offset
	*S1     //  0
}               //  8

type S3 struct { // offset
	a int64  //  0
	b int32  //  8
}                // 12

type S4 struct { // offset
	S3       //  0
	int32    // 12
}                // 16

type S5 struct {   // offset
	a [3]int32 //  0
	b int32    // 12
}                  // 16

func (S2) m() {}

func Alignof1() {
	var x int
	_ = unsafe.Alignof() // ERROR not enough arguments
	_ = unsafe.Alignof(1, 2) // ERROR too many arguments
	_ = unsafe.Alignof(int /* ERROR not an expression */)
	_ = unsafe.Alignof(42)
	_ = unsafe.Alignof(new(struct{}))
	_ = unsafe.Alignof(1<<10)
	_ = unsafe.Alignof(1 << /* ERROR constant shift overflow */ 1000)
	_ = unsafe.Alignof(nil /* ERROR "untyped nil */ )
	unsafe /* ERROR not used */ .Alignof(x)

	var y S0
	assert(unsafe.Alignof(y.a) == 1)
	assert(unsafe.Alignof(y.b) == 4)
	assert(unsafe.Alignof(y.c) == 8)
	assert(unsafe.Alignof(y.d) == 1)
	assert(unsafe.Alignof(y.e) == 8)

	var s []byte
	_ = unsafe.Alignof(s)
	_ = unsafe.Alignof(s... /* ERROR invalid use of \.\.\. */ )
}

func Alignof2() {
	f1 := func() (x int32) { return }
	f2 := func() (x, y int32) { return }
	_ = unsafe.Alignof(f0 /* ERROR used as value */ ())
	assert(unsafe.Alignof(f1()) == 4)
	_ = unsafe.Alignof(f2()) // ERROR too many arguments
}

func Offsetof1() {
	var x struct{ f int }
	_ = unsafe.Offsetof() // ERROR not enough arguments
	_ = unsafe.Offsetof(1, 2) // ERROR too many arguments
	_ = unsafe.Offsetof(int /* ERROR not a selector expression */ )
	_ = unsafe.Offsetof(x /* ERROR not a selector expression */ )
	_ = unsafe.Offsetof(nil /* ERROR not a selector expression */ )
	_ = unsafe.Offsetof(x.f)
	_ = unsafe.Offsetof((x.f))
	_ = unsafe.Offsetof((((((((x))).f)))))
	unsafe /* ERROR not used */ .Offsetof(x.f)

	var y0 S0
	assert(unsafe.Offsetof(y0.a) == 0)
	assert(unsafe.Offsetof(y0.b) == 4)
	assert(unsafe.Offsetof(y0.c) == 8)
	assert(unsafe.Offsetof(y0.d) == 16)
	assert(unsafe.Offsetof(y0.e) == 24)

	var y1 S1
	assert(unsafe.Offsetof(y1.x) == 0)
	assert(unsafe.Offsetof(y1.y) == 8)
	assert(unsafe.Offsetof(y1.z) == 24)
	assert(unsafe.Offsetof(y1.S0) == 32)

	assert(unsafe.Offsetof(y1.S0.a) == 0) // relative to S0
	assert(unsafe.Offsetof(y1.a) == 32)   // relative to S1
	assert(unsafe.Offsetof(y1.b) == 36)   // relative to S1
	assert(unsafe.Offsetof(y1.c) == 40)   // relative to S1
	assert(unsafe.Offsetof(y1.d) == 48)   // relative to S1
	assert(unsafe.Offsetof(y1.e) == 56)   // relative to S1

	var y1p *S1
	assert(unsafe.Offsetof(y1p.S0) == 32)

	type P *S1
	var p P = y1p
	assert(unsafe.Offsetof(p.S0) == 32)

	var y2 S2
	assert(unsafe.Offsetof(y2.S1) == 0)
	_ = unsafe.Offsetof(y2 /* ERROR embedded via a pointer */ .x)
	_ = unsafe.Offsetof(y2 /* ERROR method value */ .m)

	var s []byte
	_ = unsafe.Offsetof(s... /* ERROR invalid use of \.\.\. */ )
}

func Offsetof2() {
	f1 := func() (x int32) { return }
	f2 := func() (x, y int32) { return }
	_ = unsafe.Offsetof(f0 /* ERROR not a selector expression */ ())
	_ = unsafe.Offsetof(f1 /* ERROR not a selector expression */ ())
	_ = unsafe.Offsetof(f2 /* ERROR not a selector expression */ ())
}

func Sizeof1() {
	var x int
	_ = unsafe.Sizeof() // ERROR not enough arguments
	_ = unsafe.Sizeof(1, 2) // ERROR too many arguments
	_ = unsafe.Sizeof(int /* ERROR not an expression */)
	_ = unsafe.Sizeof(42)
	_ = unsafe.Sizeof(new(complex128))
	_ = unsafe.Sizeof(1<<10)
	_ = unsafe.Sizeof(1 << /* ERROR constant shift overflow */ 1000)
	_ = unsafe.Sizeof(nil /* ERROR untyped nil */ )
	unsafe /* ERROR not used */ .Sizeof(x)

	// basic types have size guarantees
	assert(unsafe.Sizeof(byte(0)) == 1)
	assert(unsafe.Sizeof(uint8(0)) == 1)
	assert(unsafe.Sizeof(int8(0)) == 1)
	assert(unsafe.Sizeof(uint16(0)) == 2)
	assert(unsafe.Sizeof(int16(0)) == 2)
	assert(unsafe.Sizeof(uint32(0)) == 4)
	assert(unsafe.Sizeof(int32(0)) == 4)
	assert(unsafe.Sizeof(float32(0)) == 4)
	assert(unsafe.Sizeof(uint64(0)) == 8)
	assert(unsafe.Sizeof(int64(0)) == 8)
	assert(unsafe.Sizeof(float64(0)) == 8)
	assert(unsafe.Sizeof(complex64(0)) == 8)
	assert(unsafe.Sizeof(complex128(0)) == 16)

	var y0 S0
	assert(unsafe.Sizeof(y0.a) == 1)
	assert(unsafe.Sizeof(y0.b) == 4)
	assert(unsafe.Sizeof(y0.c) == 8)
	assert(unsafe.Sizeof(y0.d) == 1)
	assert(unsafe.Sizeof(y0.e) == 16)
	assert(unsafe.Sizeof(y0) == 40)

	var y1 S1
	assert(unsafe.Sizeof(y1) == 72)

	var y2 S2
	assert(unsafe.Sizeof(y2) == 8)

	var y3 S3
	assert(unsafe.Sizeof(y3) == 12)

	var y4 S4
	assert(unsafe.Sizeof(y4) == 16)

	var y5 S5
	assert(unsafe.Sizeof(y5) == 16)

	var a3 [10]S3
	assert(unsafe.Sizeof(a3) == 156)

	// test case for issue 5670
	type T struct {
		a int32
		_ int32
		c int32
	}
	assert(unsafe.Sizeof(T{}) == 12)

	var s []byte
	_ = unsafe.Sizeof(s)
	_ = unsafe.Sizeof(s... /* ERROR invalid use of \.\.\. */ )
}

func Sizeof2() {
	f1 := func() (x int64) { return }
	f2 := func() (x, y int64) { return }
	_ = unsafe.Sizeof(f0 /* ERROR used as value */ ())
	assert(unsafe.Sizeof(f1()) == 8)
	_ = unsafe.Sizeof(f2()) // ERROR too many arguments
}

// self-testing only
func assert1() {
	var x int
	assert() /* ERROR not enough arguments */
	assert(1, 2) /* ERROR too many arguments */
	assert("foo" /* ERROR boolean constant */ )
	assert(x /* ERROR boolean constant */)
	assert(true)
	assert /* ERROR failed */ (false)
	_ = assert(true)

	var s []byte
	assert(s... /* ERROR invalid use of \.\.\. */ )
}

func assert2() {
	f1 := func() (x bool) { return }
	f2 := func() (x bool) { return }
	assert(f0 /* ERROR used as value */ ())
	assert(f1 /* ERROR boolean constant */ ())
	assert(f2 /* ERROR boolean constant */ ())
}

// self-testing only
func trace1() {
	// Uncomment the code below to test trace - will produce console output
	// _ = trace /* ERROR no value */ ()
	// _ = trace(1)
	// _ = trace(true, 1.2, '\'', "foo", 42i, "foo" <= "bar")

	var s []byte
	trace(s... /* ERROR invalid use of \.\.\. */ )
}

func trace2() {
	f1 := func() (x int) { return }
	f2 := func() (x int, y string) { return }
	f3 := func() (x int, y string, z []int) { return }
	_ = f1
	_ = f2
	_ = f3
	// Uncomment the code below to test trace - will produce console output
	// trace(f0())
	// trace(f1())
	// trace(f2())
	// trace(f3())
	// trace(f0(), 1)
	// trace(f1(), 1, 2)
	// trace(f2(), 1, 2, 3)
	// trace(f3(), 1, 2, 3, 4)
}
