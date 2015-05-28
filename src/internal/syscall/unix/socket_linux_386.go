// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
	"unsafe"
)

const (
	sysGETSOCKNAME = 0x6
	sysGETPEERNAME = 0x7
	sysSENDTO      = 0xb
	sysRECVFROM    = 0xc
)

func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (int, syscall.Errno)
func rawsocketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (int, syscall.Errno)

func getsockname(s int, addr []byte) error {
	l := uint32(len(addr))
	_, errno := rawsocketcall(sysGETSOCKNAME, uintptr(s), uintptr(unsafe.Pointer(&addr[0])), uintptr(unsafe.Pointer(&l)), 0, 0, 0)
	if errno != 0 {
		return error(errno)
	}
	return nil
}

func getpeername(s int, addr []byte) error {
	l := uint32(len(addr))
	_, errno := rawsocketcall(sysGETPEERNAME, uintptr(s), uintptr(unsafe.Pointer(&addr[0])), uintptr(unsafe.Pointer(&l)), 0, 0, 0)
	if errno != 0 {
		return error(errno)
	}
	return nil
}

func recvfrom(s int, b []byte, flags int, from []byte) (int, error) {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	} else {
		p = unsafe.Pointer(&emptyPayload)
	}
	l := uint32(len(from))
	n, errno := socketcall(sysRECVFROM, uintptr(s), uintptr(p), uintptr(len(b)), uintptr(flags), uintptr(unsafe.Pointer(&from[0])), uintptr(unsafe.Pointer(&l)))
	if errno != 0 {
		return int(n), error(errno)
	}
	return int(n), nil
}

func sendto(s int, b []byte, flags int, to []byte) (int, error) {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	} else {
		p = unsafe.Pointer(&emptyPayload)
	}
	n, errno := socketcall(sysSENDTO, uintptr(s), uintptr(p), uintptr(len(b)), uintptr(flags), uintptr(unsafe.Pointer(&to[0])), uintptr(len(to)))
	if errno != 0 {
		return int(n), error(errno)
	}
	return int(n), nil
}
