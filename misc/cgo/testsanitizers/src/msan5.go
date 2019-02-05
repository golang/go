// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Using reflect to set a value was not seen by msan.

/*
#include <stdlib.h>

extern void Go1(int*);
extern void Go2(char*);

// Use weak as a hack to permit defining a function even though we use export.
void C1() __attribute__ ((weak));
void C2() __attribute__ ((weak));

void C1() {
	int i;
	Go1(&i);
	if (i != 42) {
		abort();
	}
}

void C2() {
	char a[2];
	a[1] = 42;
	Go2(a);
	if (a[0] != 42) {
		abort();
	}
}
*/
import "C"

import (
	"reflect"
	"unsafe"
)

//export Go1
func Go1(p *C.int) {
	reflect.ValueOf(p).Elem().Set(reflect.ValueOf(C.int(42)))
}

//export Go2
func Go2(p *C.char) {
	a := (*[2]byte)(unsafe.Pointer(p))
	reflect.Copy(reflect.ValueOf(a[:1]), reflect.ValueOf(a[1:]))
}

func main() {
	C.C1()
	C.C2()
}
