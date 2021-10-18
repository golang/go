//go:build ignore
// +build ignore

package main

var unknown bool // defeat dead-code elimination

var p, q int

type A struct {
	f *int
	g interface{}
}

func (a A) m1() {
	print(a.f) // @pointsto command-line-arguments.p
}

func (a *A) m2() {
	print(a)   // @pointsto complit.A@struct1s:9
	print(a.f) // @pointsto command-line-arguments.p
}

type B struct {
	h *int
	A
}

func structs1() {
	b := &B{ // @line struct1s
		h: &q,
	}
	b.f = &p
	b.g = b

	print(b.h) // @pointsto command-line-arguments.q
	print(b.f) // @pointsto command-line-arguments.p
	print(b.g) // @types *B

	ptr := &b.f
	print(*ptr) // @pointsto command-line-arguments.p

	b.m1()
	b.m2()
}

// @calls command-line-arguments.structs1 -> (command-line-arguments.A).m1
// @calls command-line-arguments.structs1 -> (*command-line-arguments.A).m2
// @calls (*command-line-arguments.B).m1 -> (command-line-arguments.A).m1
// @calls (*command-line-arguments.B).m2 -> (*command-line-arguments.A).m2

type T struct {
	x int
	y int
}

type S struct {
	a [3]T
	b *[3]T
	c [3]*T
}

func structs2() {
	var s S          // @line s2s
	print(&s)        // @pointsto s@s2s:6
	print(&s.a)      // @pointsto s.a@s2s:6
	print(&s.a[0])   // @pointsto s.a[*]@s2s:6
	print(&s.a[0].x) // @pointsto s.a[*].x@s2s:6
	print(&s.a[0].y) // @pointsto s.a[*].y@s2s:6
	print(&s.b)      // @pointsto s.b@s2s:6
	print(&s.b[0])   // @pointsto
	print(&s.b[0].x) // @pointsto
	print(&s.b[0].y) // @pointsto
	print(&s.c)      // @pointsto s.c@s2s:6
	print(&s.c[0])   // @pointsto s.c[*]@s2s:6
	print(&s.c[0].x) // @pointsto
	print(&s.c[0].y) // @pointsto

	var s2 S          // @line s2s2
	s2.b = new([3]T)  // @line s2s2b
	print(s2.b)       // @pointsto new@s2s2b:12
	print(&s2.b)      // @pointsto s2.b@s2s2:6
	print(&s2.b[0])   // @pointsto new[*]@s2s2b:12
	print(&s2.b[0].x) // @pointsto new[*].x@s2s2b:12
	print(&s2.b[0].y) // @pointsto new[*].y@s2s2b:12
	print(&s2.c[0].x) // @pointsto
	print(&s2.c[0].y) // @pointsto

	var s3 S          // @line s2s3
	s3.c[2] = new(T)  // @line s2s3c
	print(&s3.c)      // @pointsto s3.c@s2s3:6
	print(s3.c[1])    // @pointsto new@s2s3c:15
	print(&s3.c[1])   // @pointsto s3.c[*]@s2s3:6
	print(&s3.c[1].x) // @pointsto new.x@s2s3c:15
	print(&s3.c[1].y) // @pointsto new.y@s2s3c:15
}

func main() {
	structs1()
	structs2()
}
