// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interface values containing structures.

package main

import "os"

var fail int

func check(b bool, msg string) {
	if (!b) {
		println("failure in", msg)
		fail++
	}
}

type I1 interface { Get() int; Put(int) }

type S1 struct { i int }
func (p S1) Get() int { return p.i }
func (p S1) Put(i int) { p.i = i }

func f1() {
	s := S1{1}
	var i I1 = s
	i.Put(2)
	check(i.Get() == 1, "f1 i")
	check(s.i == 1, "f1 s")
}

func f2() {
	s := S1{1}
	var i I1 = &s
	i.Put(2)
	check(i.Get() == 1, "f2 i")
	check(s.i == 1, "f2 s")
}

func f3() {
	s := &S1{1}
	var i I1 = s
	i.Put(2)
	check(i.Get() == 1, "f3 i")
	check(s.i == 1, "f3 s")
}

type S2 struct { i int }
func (p *S2) Get() int { return p.i }
func (p *S2) Put(i int) { p.i = i }

// Disallowed by restriction of values going to pointer receivers
// func f4() {
//	 s := S2{1}
//	 var i I1 = s
//	 i.Put(2)
//	 check(i.Get() == 2, "f4 i")
//	 check(s.i == 1, "f4 s")
// }

func f5() {
	s := S2{1}
	var i I1 = &s
	i.Put(2)
	check(i.Get() == 2, "f5 i")
	check(s.i == 2, "f5 s")
}

func f6() {
	s := &S2{1}
	var i I1 = s
	i.Put(2)
	check(i.Get() == 2, "f6 i")
	check(s.i == 2, "f6 s")
}

type I2 interface { Get() int64; Put(int64) }

type S3 struct { i, j, k, l int64 }
func (p S3) Get() int64 { return p.l }
func (p S3) Put(i int64) { p.l = i }

func f7() {
	s := S3{1, 2, 3, 4}
	var i I2 = s
	i.Put(5)
	check(i.Get() == 4, "f7 i")
	check(s.l == 4, "f7 s")
}

func f8() {
	s := S3{1, 2, 3, 4}
	var i I2 = &s
	i.Put(5)
	check(i.Get() == 4, "f8 i")
	check(s.l == 4, "f8 s")
}

func f9() {
	s := &S3{1, 2, 3, 4}
	var i I2 = s
	i.Put(5)
	check(i.Get() == 4, "f9 i")
	check(s.l == 4, "f9 s")
}

type S4 struct { i, j, k, l int64 }
func (p *S4) Get() int64 { return p.l }
func (p *S4) Put(i int64) { p.l = i }

// Disallowed by restriction of values going to pointer receivers
// func f10() {
//	 s := S4{1, 2, 3, 4}
//	 var i I2 = s
//	 i.Put(5)
//	 check(i.Get() == 5, "f10 i")
//	 check(s.l == 4, "f10 s")
// }

func f11() {
	s := S4{1, 2, 3, 4}
	var i I2 = &s
	i.Put(5)
	check(i.Get() == 5, "f11 i")
	check(s.l == 5, "f11 s")
}

func f12() {
	s := &S4{1, 2, 3, 4}
	var i I2 = s
	i.Put(5)
	check(i.Get() == 5, "f12 i")
	check(s.l == 5, "f12 s")
}

func main() {
	f1()
	f2()
	f3()
//	f4()
	f5()
	f6()
	f7()
	f8()
	f9()
//	f10()
	f11()
	f12()
	if fail > 0 {
		os.Exit(1)
	}
}
