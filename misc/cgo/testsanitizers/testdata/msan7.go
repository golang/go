// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test passing C struct to exported Go function.

/*
#include <stdint.h>
#include <stdlib.h>

// T is a C struct with alignment padding after b.
// The padding bytes are not considered initialized by MSAN.
// It is big enough to be passed on stack in C ABI (and least
// on AMD64).
typedef struct { char b; uintptr_t x, y; } T;

extern void F(T);

// Use weak as a hack to permit defining a function even though we use export.
void CF(int x) __attribute__ ((weak));
void CF(int x) {
	T *t = malloc(sizeof(T));
	t->b = (char)x;
	t->x = x;
	t->y = x;
	F(*t);
}
*/
import "C"

//export F
func F(t C.T) { println(t.b, t.x, t.y) }

func main() {
	C.CF(C.int(0))
}
