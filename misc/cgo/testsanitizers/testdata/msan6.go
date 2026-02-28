// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// A C function returning a value on the Go stack could leave the Go
// stack marked as uninitialized, potentially causing a later error
// when the stack is used for something else. Issue 26209.

/*
#cgo LDFLAGS: -fsanitize=memory
#cgo CPPFLAGS: -fsanitize=memory

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	uintptr_t a[20];
} S;

S f() {
	S *p;

	p = (S *)(malloc(sizeof(S)));
	p->a[0] = 0;
	return *p;
}
*/
import "C"

// allocateStack extends the stack so that stack copying doesn't
// confuse the msan data structures.
//go:noinline
func allocateStack(i int) int {
	if i == 0 {
		return i
	}
	return allocateStack(i - 1)
}

// F1 marks a chunk of stack as uninitialized.
// C.f returns an uninitialized struct on the stack, so msan will mark
// the stack as uninitialized.
//go:noinline
func F1() uintptr {
	s := C.f()
	return uintptr(s.a[0])
}

// F2 allocates a struct on the stack and converts it to an empty interface,
// which will call msanread and see that the data appears uninitialized.
//go:noinline
func F2() interface{} {
	return C.S{}
}

func poisonStack(i int) int {
	if i == 0 {
		return int(F1())
	}
	F1()
	r := poisonStack(i - 1)
	F2()
	return r
}

func main() {
	allocateStack(16384)
	poisonStack(128)
}
