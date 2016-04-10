// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Basic test cases for cgo.

package cgotest

/*
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

#define SHIFT(x, y)  ((x)<<(y))
#define KILO SHIFT(1, 10)
#define UINT32VAL 0xc008427bU

enum E {
	Enum1 = 1,
	Enum2 = 2,
};

typedef unsigned char cgo_uuid_t[20];

void uuid_generate(cgo_uuid_t x) {
	x[0] = 0;
}

struct S {
	int x;
};

extern enum E myConstFunc(struct S* const ctx, int const id, struct S **const filter);

enum E myConstFunc(struct S *const ctx, int const id, struct S **const filter) { return 0; }

// issue 1222
typedef union {
	long align;
} xxpthread_mutex_t;

struct ibv_async_event {
	union {
		int x;
	} element;
};

struct ibv_context {
	xxpthread_mutex_t mutex;
};

int add(int x, int y) {
	return x+y;
};
*/
import "C"
import (
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

const EINVAL = C.EINVAL /* test #define */

var KILO = C.KILO

func uuidgen() {
	var uuid C.cgo_uuid_t
	C.uuid_generate(&uuid[0])
}

func Strtol(s string, base int) (int, error) {
	p := C.CString(s)
	n, err := C.strtol(p, nil, C.int(base))
	C.free(unsafe.Pointer(p))
	return int(n), err
}

func Atol(s string) int {
	p := C.CString(s)
	n := C.atol(p)
	C.free(unsafe.Pointer(p))
	return int(n)
}

func testConst(t *testing.T) {
	C.myConstFunc(nil, 0, nil)
}

func testEnum(t *testing.T) {
	if C.Enum1 != 1 || C.Enum2 != 2 {
		t.Error("bad enum", C.Enum1, C.Enum2)
	}
}

func testAtol(t *testing.T) {
	l := Atol("123")
	if l != 123 {
		t.Error("Atol 123: ", l)
	}
}

func testErrno(t *testing.T) {
	p := C.CString("no-such-file")
	m := C.CString("r")
	f, err := C.fopen(p, m)
	C.free(unsafe.Pointer(p))
	C.free(unsafe.Pointer(m))
	if err == nil {
		C.fclose(f)
		t.Fatalf("C.fopen: should fail")
	}
	if err != syscall.ENOENT {
		t.Fatalf("C.fopen: unexpected error: %v", err)
	}
}

func testMultipleAssign(t *testing.T) {
	p := C.CString("234")
	n, m := C.strtol(p, nil, 345), C.strtol(p, nil, 10)
	if runtime.GOOS == "openbsd" {
		// Bug in OpenBSD strtol(3) - base > 36 succeeds.
		if (n != 0 && n != 239089) || m != 234 {
			t.Fatal("Strtol x2: ", n, m)
		}
	} else if n != 0 || m != 234 {
		t.Fatal("Strtol x2: ", n, m)
	}
	C.free(unsafe.Pointer(p))
}

var (
	cuint  = (C.uint)(0)
	culong C.ulong
	cchar  C.char
)

type Context struct {
	ctx *C.struct_ibv_context
}

func benchCgoCall(b *testing.B) {
	const x = C.int(2)
	const y = C.int(3)
	for i := 0; i < b.N; i++ {
		C.add(x, y)
	}
}

// Issue 2470.
func testUnsignedInt(t *testing.T) {
	a := (int64)(C.UINT32VAL)
	b := (int64)(0xc008427b)
	if a != b {
		t.Errorf("Incorrect unsigned int - got %x, want %x", a, b)
	}
}

// Static (build-time) test that syntax traversal visits all operands of s[i:j:k].
func sliceOperands(array [2000]int) {
	_ = array[C.KILO:C.KILO:C.KILO] // no type error
}
