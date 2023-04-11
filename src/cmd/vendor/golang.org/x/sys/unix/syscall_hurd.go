// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build hurd
// +build hurd

package unix

/*
#include <stdint.h>
int ioctl(int, unsigned long int, uintptr_t);
*/
import "C"

func ioctl(fd int, req uint, arg uintptr) (err error) {
	r0, er := C.ioctl(C.int(fd), C.ulong(req), C.uintptr_t(arg))
	if r0 == -1 && er != nil {
		err = er
	}
	return
}

func ioctlPtr(fd int, req uint, arg unsafe.Pointer) (err error) {
	r0, er := C.ioctl(C.int(fd), C.ulong(req), C.uintptr_t(uintptr(arg)))
	if r0 == -1 && er != nil {
		err = er
	}
	return
}
