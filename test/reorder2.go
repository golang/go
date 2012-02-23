// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test reorderings; derived from fixedbugs/bug294.go.

package main

var log string

type TT int

func (t TT) a(s string) TT {
	log += "a(" + s + ")"
	return t
}

func (TT) b(s string) string {
	log += "b(" + s + ")"
	return s
}

type F func(s string) F

func a(s string) F {
	log += "a(" + s + ")"
	return F(a)
}

func b(s string) string {
	log += "b(" + s + ")"
	return s
}

type I interface {
	a(s string) I
	b(s string) string
}

type T1 int

func (t T1) a(s string) I {
	log += "a(" + s + ")"
	return t
}

func (T1) b(s string) string {
	log += "b(" + s + ")"
	return s
}

// f(g(), h()) where g is not inlinable but h is will have the same problem.
// As will x := g() + h() (same conditions).
// And g() <- h().
func f(x, y string) {
	log += "f(" + x + ", " + y + ")"
}

func ff(x, y string) {
	for false {
	} // prevent inl
	log += "ff(" + x + ", " + y + ")"
}

func h(x string) string {
	log += "h(" + x + ")"
	return x
}

func g(x string) string {
	for false {
	} // prevent inl
	log += "g(" + x + ")"
	return x
}

func main() {
	err := 0
	var t TT
	if a("1")("2")("3"); log != "a(1)a(2)a(3)" {
		println("expecting a(1)a(2)a(3) , got ", log)
		err++
	}
	log = ""

	if t.a("1").a(t.b("2")); log != "a(1)b(2)a(2)" {
		println("expecting a(1)b(2)a(2), got ", log)
		err++
	}
	log = ""
	if a("3")(b("4"))(b("5")); log != "a(3)b(4)a(4)b(5)a(5)" {
		println("expecting a(3)b(4)a(4)b(5)a(5), got ", log)
		err++
	}
	log = ""
	var i I = T1(0)
	if i.a("6").a(i.b("7")).a(i.b("8")).a(i.b("9")); log != "a(6)b(7)a(7)b(8)a(8)b(9)a(9)" {
		println("expecting a(6)ba(7)ba(8)ba(9), got", log)
		err++
	}
	log = ""

	if s := t.a("1").b("3"); log != "a(1)b(3)" || s != "3" {
		println("expecting a(1)b(3) and 3, got ", log, " and ", s)
		err++
	}
	log = ""

	if s := t.a("1").a(t.b("2")).b("3") + t.a("4").b("5"); log != "a(1)b(2)a(2)b(3)a(4)b(5)" || s != "35" {
		println("expecting a(1)b(2)a(2)b(3)a(4)b(5) and 35, got ", log, " and ", s)
		err++
	}
	log = ""

	if s := t.a("4").b("5") + t.a("1").a(t.b("2")).b("3"); log != "a(4)b(5)a(1)b(2)a(2)b(3)" || s != "53" {
		println("expecting a(4)b(5)a(1)b(2)a(2)b(3) and 35, got ", log, " and ", s)
		err++
	}
	log = ""

	if ff(g("1"), g("2")); log != "g(1)g(2)ff(1, 2)" {
		println("expecting g(1)g(2)ff..., got ", log)
		err++
	}
	log = ""

	if ff(g("1"), h("2")); log != "g(1)h(2)ff(1, 2)" {
		println("expecting g(1)h(2)ff..., got ", log)
		err++
	}
	log = ""

	if ff(h("1"), g("2")); log != "h(1)g(2)ff(1, 2)" {
		println("expecting h(1)g(2)ff..., got ", log)
		err++
	}
	log = ""

	if ff(h("1"), h("2")); log != "h(1)h(2)ff(1, 2)" {
		println("expecting h(1)h(2)ff..., got ", log)
		err++
	}
	log = ""

	if s := g("1") + g("2"); log != "g(1)g(2)" || s != "12" {
		println("expecting g1g2 and 12, got ", log, " and ", s)
		err++
	}
	log = ""

	if s := g("1") + h("2"); log != "g(1)h(2)" || s != "12" {
		println("expecting g1h2 and 12, got ", log, " and ", s)
		err++
	}
	log = ""

	if s := h("1") + g("2"); log != "h(1)g(2)" || s != "12" {
		println("expecting h1g2 and 12, got ", log, " and ", s)
		err++
	}
	log = ""

	if s := h("1") + h("2"); log != "h(1)h(2)" || s != "12" {
		println("expecting h1h2 and 12, got ", log, " and ", s)
		err++
	}
	log = ""

	if err > 0 {
		panic("fail")
	}
}
