// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4654.
// Check error for conversion and 'not used' in defer/go.

package p

import "unsafe"

func f() {
	defer int(0) // ERROR "defer requires function call, not conversion|is not used"
	go string([]byte("abc")) // ERROR "go requires function call, not conversion|is not used"
	
	var c complex128
	var f float64
	var t struct {X int}

	var x []int
	defer append(x, 1) // ERROR "defer discards result of append|is not used"
	defer cap(x) // ERROR "defer discards result of cap|is not used"
	defer complex(1, 2) // ERROR "defer discards result of complex|is not used"
	defer complex(f, 1) // ERROR "defer discards result of complex|is not used"
	defer imag(1i) // ERROR "defer discards result of imag|is not used"
	defer imag(c) // ERROR "defer discards result of imag|is not used"
	defer len(x) // ERROR "defer discards result of len|is not used"
	defer make([]int, 1) // ERROR "defer discards result of make|is not used"
	defer make(chan bool) // ERROR "defer discards result of make|is not used"
	defer make(map[string]int) // ERROR "defer discards result of make|is not used"
	defer new(int) // ERROR "defer discards result of new|is not used"
	defer real(1i) // ERROR "defer discards result of real|is not used"
	defer real(c) // ERROR "defer discards result of real|is not used"
	defer append(x, 1) // ERROR "defer discards result of append|is not used"
	defer append(x, 1) // ERROR "defer discards result of append|is not used"
	defer unsafe.Alignof(t.X) // ERROR "defer discards result of unsafe.Alignof|is not used"
	defer unsafe.Offsetof(t.X) // ERROR "defer discards result of unsafe.Offsetof|is not used"
	defer unsafe.Sizeof(t) // ERROR "defer discards result of unsafe.Sizeof|is not used"
	
	defer copy(x, x) // ok
	m := make(map[int]int)
	defer delete(m, 1) // ok
	defer panic(1) // ok
	defer print(1) // ok
	defer println(1) // ok
	defer recover() // ok

	int(0) // ERROR "int\(0\) evaluated but not used|is not used"
	string([]byte("abc")) // ERROR "string\(.*\) evaluated but not used|is not used"

	append(x, 1) // ERROR "not used"
	cap(x) // ERROR "not used"
	complex(1, 2) // ERROR "not used"
	complex(f, 1) // ERROR "not used"
	imag(1i) // ERROR "not used"
	imag(c) // ERROR "not used"
	len(x) // ERROR "not used"
	make([]int, 1) // ERROR "not used"
	make(chan bool) // ERROR "not used"
	make(map[string]int) // ERROR "not used"
	new(int) // ERROR "not used"
	real(1i) // ERROR "not used"
	real(c) // ERROR "not used"
	append(x, 1) // ERROR "not used"
	append(x, 1) // ERROR "not used"
	unsafe.Alignof(t.X) // ERROR "not used"
	unsafe.Offsetof(t.X) // ERROR "not used"
	unsafe.Sizeof(t) // ERROR "not used"
}
