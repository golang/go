// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo

package net

/*
#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
*/
import "C"

import "unsafe"

const cgoAddrInfoFlags = C.AI_CANONNAME

func cgoNameinfoPTR(b []byte, sa *C.struct_sockaddr, salen C.socklen_t) (int, error) {
	gerrno, err := C.getnameinfo(sa, C.size_t(salen), (*C.char)(unsafe.Pointer(&b[0])), C.size_t(len(b)), nil, 0, C.NI_NAMEREQD)
	return int(gerrno), err
}
