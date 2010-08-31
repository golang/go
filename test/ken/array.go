// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func setpd(a []int) {
	//	print("setpd a=", a, " len=", len(a), " cap=", cap(a), "\n");
	for i := 0; i < len(a); i++ {
		a[i] = i
	}
}

func sumpd(a []int) int {
	//	print("sumpd a=", a, " len=", len(a), " cap=", cap(a), "\n");
	t := 0
	for i := 0; i < len(a); i++ {
		t += a[i]
	}
	//	print("sumpd t=", t, "\n");
	return t
}

func setpf(a *[20]int) {
	//	print("setpf a=", a, " len=", len(a), " cap=", cap(a), "\n");
	for i := 0; i < len(a); i++ {
		a[i] = i
	}
}

func sumpf(a *[20]int) int {
	//	print("sumpf a=", a, " len=", len(a), " cap=", cap(a), "\n");
	t := 0
	for i := 0; i < len(a); i++ {
		t += a[i]
	}
	//	print("sumpf t=", t, "\n");
	return t
}

func res(t int, lb, hb int) {
	sb := (hb - lb) * (hb + lb - 1) / 2
	if t != sb {
		print("lb=", lb,
			"; hb=", hb,
			"; t=", t,
			"; sb=", sb,
			"\n")
		panic("res")
	}
}

// call ptr dynamic with ptr dynamic
func testpdpd() {
	a := make([]int, 10, 100)
	if len(a) != 10 && cap(a) != 100 {
		print("len and cap from new: ", len(a), " ", cap(a), "\n")
		panic("fail")
	}

	a = a[0:100]
	setpd(a)

	a = a[0:10]
	res(sumpd(a), 0, 10)

	a = a[5:25]
	res(sumpd(a), 5, 25)
}

// call ptr fixed with ptr fixed
func testpfpf() {
	var a [20]int

	setpf(&a)
	res(sumpf(&a), 0, 20)
}

// call ptr dynamic with ptr fixed from new
func testpdpf1() {
	a := new([40]int)
	setpd(a[0:])
	res(sumpd(a[0:]), 0, 40)

	b := (*a)[5:30]
	res(sumpd(b), 5, 30)
}

// call ptr dynamic with ptr fixed from var
func testpdpf2() {
	var a [80]int

	setpd(a[0:])
	res(sumpd(a[0:]), 0, 80)
}

// generate bounds error with ptr dynamic
func testpdfault() {
	a := make([]int, 100)

	print("good\n")
	for i := 0; i < 100; i++ {
		a[i] = 0
	}
	print("should fault\n")
	a[100] = 0
	print("bad\n")
}

// generate bounds error with ptr fixed
func testfdfault() {
	var a [80]int

	print("good\n")
	for i := 0; i < 80; i++ {
		a[i] = 0
	}
	print("should fault\n")
	x := 80
	a[x] = 0
	print("bad\n")
}

func main() {
	testpdpd()
	testpfpf()
	testpdpf1()
	testpdpf2()
	//	print("testpdfault\n");	testpdfault();
	//	print("testfdfault\n");	testfdfault();
}
