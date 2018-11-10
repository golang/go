// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7786. No runtime test, just make sure that typedef and struct/union/class are interchangeable at compile time.

package cgotest

// struct test7786;
// typedef struct test7786 typedef_test7786;
// void f7786(struct test7786 *ctx) {}
// void g7786(typedef_test7786 *ctx) {}
//
// typedef struct body7786 typedef_body7786;
// struct body7786 { int x; };
// void b7786(struct body7786 *ctx) {}
// void c7786(typedef_body7786 *ctx) {}
//
// typedef union union7786 typedef_union7786;
// void u7786(union union7786 *ctx) {}
// void v7786(typedef_union7786 *ctx) {}
import "C"

func f() {
	var x1 *C.typedef_test7786
	var x2 *C.struct_test7786
	x1 = x2
	x2 = x1
	C.f7786(x1)
	C.f7786(x2)
	C.g7786(x1)
	C.g7786(x2)

	var b1 *C.typedef_body7786
	var b2 *C.struct_body7786
	b1 = b2
	b2 = b1
	C.b7786(b1)
	C.b7786(b2)
	C.c7786(b1)
	C.c7786(b2)

	var u1 *C.typedef_union7786
	var u2 *C.union_union7786
	u1 = u2
	u2 = u1
	C.u7786(u1)
	C.u7786(u2)
	C.v7786(u1)
	C.v7786(u2)
}
