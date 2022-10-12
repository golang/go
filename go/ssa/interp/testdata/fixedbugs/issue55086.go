// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func a() (r string) {
	s := "initial"
	var p *struct{ i int }
	defer func() {
		recover()
		r = s
	}()

	s, p.i = "set", 2 // s must be set before p.i panics
	return "unreachable"
}

func b() (r string) {
	s := "initial"
	fn := func() []int { panic("") }
	defer func() {
		recover()
		r = s
	}()

	s, fn()[0] = "set", 2 // fn() panics before any assignment occurs
	return "unreachable"
}

func c() (r string) {
	s := "initial"
	var p map[int]int
	defer func() {
		recover()
		r = s
	}()

	s, p[0] = "set", 2 //s must be set before p[0] index panics"
	return "unreachable"
}

func d() (r string) {
	s := "initial"
	var p map[int]int
	defer func() {
		recover()
		r = s
	}()
	fn := func() int { panic("") }

	s, p[0] = "set", fn() // fn() panics before s is set
	return "unreachable"
}

func e() (r string) {
	s := "initial"
	p := map[int]int{}
	defer func() {
		recover()
		r = s
	}()
	fn := func() int { panic("") }

	s, p[fn()] = "set", 0 // fn() panics before any assignment occurs
	return "unreachable"
}

func f() (r string) {
	s := "initial"
	p := []int{}
	defer func() {
		recover()
		r = s
	}()

	s, p[1] = "set", 0 // p[1] panics after s is set
	return "unreachable"
}

func g() (r string) {
	s := "initial"
	p := map[any]any{}
	defer func() {
		recover()
		r = s
	}()
	var i any = func() {}
	s, p[i] = "set", 0 // p[i] panics after s is set
	return "unreachable"
}

func h() (r string) {
	fail := false
	defer func() {
		recover()
		if fail {
			r = "fail"
		} else {
			r = "success"
		}
	}()

	type T struct{ f int }
	var p *struct{ *T }

	// The implicit "p.T" operand should be evaluated in phase 1 (and panic),
	// before the "fail = true" assignment in phase 2.
	fail, p.f = true, 0
	return "unreachable"
}

func main() {
	for _, test := range []struct {
		fn   func() string
		want string
		desc string
	}{
		{a, "set", "s must be set before p.i panics"},
		{b, "initial", "p() panics before s is set"},
		{c, "set", "s must be set before p[0] index panics"},
		{d, "initial", "fn() panics before s is set"},
		{e, "initial", "fn() panics before s is set"},
		{f, "set", "p[1] panics after s is set"},
		{g, "set", "p[i] panics after s is set"},
		{h, "success", "p.T panics before fail is set"},
	} {
		if test.fn() != test.want {
			panic(test.desc)
		}
	}
}
