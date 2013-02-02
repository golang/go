// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4463: test builtin functions in statement context and in
// go/defer functions.

package p

import "unsafe"

func F() {
	var a []int
	var c chan int
	var m map[int]int
	var s struct{ f int }

	append(a, 0)			// ERROR "not used"
	cap(a)				// ERROR "not used"
	complex(1, 2)			// ERROR "not used"
	imag(1i)			// ERROR "not used"
	len(a)				// ERROR "not used"
	make([]int, 10)			// ERROR "not used"
	new(int)			// ERROR "not used"
	real(1i)			// ERROR "not used"
	unsafe.Alignof(a)		// ERROR "not used"
	unsafe.Offsetof(s.f)		// ERROR "not used"
	unsafe.Sizeof(a)		// ERROR "not used"

	close(c)
	copy(a, a)
	delete(m, 0)
	panic(0)
	print("foo")
	println("bar")
	recover()

	(close(c))
	(copy(a, a))
	(delete(m, 0))
	(panic(0))
	(print("foo"))
	(println("bar"))
	(recover())

	go append(a, 0)			// ERROR "discards result"
	go cap(a)			// ERROR "discards result"
	go complex(1, 2)		// ERROR "discards result"
	go imag(1i)			// ERROR "discards result"
	go len(a)			// ERROR "discards result"
	go make([]int, 10)		// ERROR "discards result"
	go new(int)			// ERROR "discards result"
	go real(1i)			// ERROR "discards result"
	go unsafe.Alignof(a)		// ERROR "discards result"
	go unsafe.Offsetof(s.f)		// ERROR "discards result"
	go unsafe.Sizeof(a)		// ERROR "discards result"

	go close(c)
	go copy(a, a)
	go delete(m, 0)
	go panic(0)
	go print("foo")
	go println("bar")
	go recover()

	defer append(a, 0)		// ERROR "discards result"
	defer cap(a)			// ERROR "discards result"
	defer complex(1, 2)		// ERROR "discards result"
	defer imag(1i)			// ERROR "discards result"
	defer len(a)			// ERROR "discards result"
	defer make([]int, 10)		// ERROR "discards result"
	defer new(int)			// ERROR "discards result"
	defer real(1i)			// ERROR "discards result"
	defer unsafe.Alignof(a)		// ERROR "discards result"
	defer unsafe.Offsetof(s.f)	// ERROR "discards result"
	defer unsafe.Sizeof(a)		// ERROR "discards result"

	defer close(c)
	defer copy(a, a)
	defer delete(m, 0)
	defer panic(0)
	defer print("foo")
	defer println("bar")
	defer recover()
}
