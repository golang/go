// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

const C = 1

var x, x1, x2 int
var b bool
var s string
var c chan int
var cp complex128
var slice []int
var array [2]int
var bytes []byte
var runes []rune
var r rune

func f0()            {}
func f1() int        { return 1 }
func f2() (int, int) { return 1, 1 }

type T struct{ X int }

func (T) M1() int { return 1 }
func (T) M0()     {}
func (T) M()      {}

var t T
var tp *T

type I interface{ M() }

var i I

var m map[int]int

func _() {
	// Note: if the next line changes to x, the error silences the x+x etc below!
	x1 // ERROR "x1 evaluated but not used"

	nil                    // ERROR "nil evaluated but not used"
	C                      // ERROR  "C evaluated but not used"
	1                      // ERROR "1 evaluated but not used"
	x + x                  // ERROR "x \+ x evaluated but not used"
	x - x                  // ERROR "x - x evaluated but not used"
	x | x                  // ERROR "x \| x evaluated but not used"
	"a" + s                // ERROR ".a. \+ s evaluated but not used"
	&x                     // ERROR "&x evaluated but not used"
	b && b                 // ERROR "b && b evaluated but not used"
	append(slice, 1)       // ERROR "append\(slice, 1\) evaluated but not used"
	string(bytes)          // ERROR "string\(bytes\) evaluated but not used"
	string(runes)          // ERROR "string\(runes\) evaluated but not used"
	f0()                   // ok
	f1()                   // ok
	f2()                   // ok
	_ = f0()               // ERROR "f0\(\) used as value"
	_ = f1()               // ok
	_, _ = f2()            // ok
	_ = f2()               // ERROR "assignment mismatch: 1 variable but f2 returns 2 values"
	_ = f1(), 0            // ERROR "assignment mismatch: 1 variable but 2 values"
	T.M0                   // ERROR "T.M0 evaluated but not used"
	t.M0                   // ERROR "t.M0 evaluated but not used"
	cap                    // ERROR "use of builtin cap not in function call"
	cap(slice)             // ERROR "cap\(slice\) evaluated but not used"
	close(c)               // ok
	_ = close(c)           // ERROR "close\(c\) used as value"
	func() {}              // ERROR "func literal evaluated but not used"
	X{}                    // ERROR "undefined: X"
	map[string]int{}       // ERROR "map\[string\]int{} evaluated but not used"
	struct{}{}             // ERROR "struct ?{}{} evaluated but not used"
	[1]int{}               // ERROR "\[1\]int{} evaluated but not used"
	[]int{}                // ERROR "\[\]int{} evaluated but not used"
	&struct{}{}            // ERROR "&struct ?{}{} evaluated but not used"
	float32(x)             // ERROR "float32\(x\) evaluated but not used"
	I(t)                   // ERROR "I\(t\) evaluated but not used"
	int(x)                 // ERROR "int\(x\) evaluated but not used"
	copy(slice, slice)     // ok
	_ = copy(slice, slice) // ok
	delete(m, 1)           // ok
	_ = delete(m, 1)       // ERROR "delete\(m, 1\) used as value"
	t.X                    // ERROR "t.X evaluated but not used"
	tp.X                   // ERROR "tp.X evaluated but not used"
	t.M                    // ERROR "t.M evaluated but not used"
	I.M                    // ERROR "I.M evaluated but not used"
	i.(T)                  // ERROR "i.\(T\) evaluated but not used"
	x == x                 // ERROR "x == x evaluated but not used"
	x != x                 // ERROR "x != x evaluated but not used"
	x != x                 // ERROR "x != x evaluated but not used"
	x < x                  // ERROR "x < x evaluated but not used"
	x >= x                 // ERROR "x >= x evaluated but not used"
	x > x                  // ERROR "x > x evaluated but not used"
	*tp                    // ERROR "\*tp evaluated but not used"
	slice[0]               // ERROR "slice\[0\] evaluated but not used"
	m[1]                   // ERROR "m\[1\] evaluated but not used"
	len(slice)             // ERROR "len\(slice\) evaluated but not used"
	make(chan int)         // ERROR "make\(chan int\) evaluated but not used"
	make(map[int]int)      // ERROR "make\(map\[int\]int\) evaluated but not used"
	make([]int, 1)         // ERROR "make\(\[\]int, 1\) evaluated but not used"
	x * x                  // ERROR "x \* x evaluated but not used"
	x / x                  // ERROR "x / x evaluated but not used"
	x % x                  // ERROR "x % x evaluated but not used"
	x << x                 // ERROR "x << x evaluated but not used"
	x >> x                 // ERROR "x >> x evaluated but not used"
	x & x                  // ERROR "x & x evaluated but not used"
	x &^ x                 // ERROR "x &\^ x evaluated but not used"
	new(int)               // ERROR "new\(int\) evaluated but not used"
	!b                     // ERROR "!b evaluated but not used"
	^x                     // ERROR "\^x evaluated but not used"
	+x                     // ERROR "\+x evaluated but not used"
	-x                     // ERROR "-x evaluated but not used"
	b || b                 // ERROR "b \|\| b evaluated but not used"
	panic(1)               // ok
	_ = panic(1)           // ERROR "panic\(1\) used as value"
	print(1)               // ok
	_ = print(1)           // ERROR "print\(1\) used as value"
	println(1)             // ok
	_ = println(1)         // ERROR "println\(1\) used as value"
	c <- 1                 // ok
	slice[1:1]             // ERROR "slice\[1:1\] evaluated but not used"
	array[1:1]             // ERROR "array\[1:1\] evaluated but not used"
	s[1:1]                 // ERROR "s\[1:1\] evaluated but not used"
	slice[1:1:1]           // ERROR "slice\[1:1:1\] evaluated but not used"
	array[1:1:1]           // ERROR "array\[1:1:1\] evaluated but not used"
	recover()              // ok
	<-c                    // ok
	string(r)              // ERROR "string\(r\) evaluated but not used"
	iota                   // ERROR "undefined: iota"
	real(cp)               // ERROR "real\(cp\) evaluated but not used"
	imag(cp)               // ERROR "imag\(cp\) evaluated but not used"
	complex(1, 2)          // ERROR "complex\(1, 2\) evaluated but not used"
	unsafe.Alignof(t.X)    // ERROR "unsafe.Alignof\(t.X\) evaluated but not used"
	unsafe.Offsetof(t.X)   // ERROR "unsafe.Offsetof\(t.X\) evaluated but not used"
	unsafe.Sizeof(t)       // ERROR "unsafe.Sizeof\(t\) evaluated but not used"
	_ = int                // ERROR "type int is not an expression"
	(x)                    // ERROR "x evaluated but not used"
	_ = new(x2)            // ERROR "x2 is not a type"
	_ = new(1 + 1)         // ERROR "1 \+ 1 is not a type"
}
