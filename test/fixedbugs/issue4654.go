// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4654.
// Check error for conversion and 'not used' in defer/go.

package p

import "unsafe"

func f() {
	defer int(0) // ERROR "defer requires function call, not conversion"
	go string([]byte("abc")) // ERROR "go requires function call, not conversion"
	
	var c complex128
	var f float64
	var t struct {X int}

	var x []int
	defer append(x, 1) // ERROR "defer discards result of append"
	defer cap(x) // ERROR "defer discards result of cap"
	defer complex(1, 2) // ERROR "defer discards result of complex"
	defer complex(f, 1) // ERROR "defer discards result of complex"
	defer imag(1i) // ERROR "defer discards result of imag"
	defer imag(c) // ERROR "defer discards result of imag"
	defer len(x) // ERROR "defer discards result of len"
	defer make([]int, 1) // ERROR "defer discards result of make"
	defer make(chan bool) // ERROR "defer discards result of make"
	defer make(map[string]int) // ERROR "defer discards result of make"
	defer new(int) // ERROR "defer discards result of new"
	defer real(1i) // ERROR "defer discards result of real"
	defer real(c) // ERROR "defer discards result of real"
	defer append(x, 1) // ERROR "defer discards result of append"
	defer append(x, 1) // ERROR "defer discards result of append"
	defer unsafe.Alignof(t.X) // ERROR "defer discards result of unsafe.Alignof"
	defer unsafe.Offsetof(t.X) // ERROR "defer discards result of unsafe.Offsetof"
	defer unsafe.Sizeof(t) // ERROR "defer discards result of unsafe.Sizeof"
	
	defer copy(x, x) // ok
	m := make(map[int]int)
	defer delete(m, 1) // ok
	defer panic(1) // ok
	defer print(1) // ok
	defer println(1) // ok
	defer recover() // ok

	int(0) // ERROR "int\(0\) evaluated but not used"
	string([]byte("abc")) // ERROR "string\(.*\) evaluated but not used"

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
