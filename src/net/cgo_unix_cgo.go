// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !netgo && unix && !darwin

package net

/*
#define _GNU_SOURCE 1

#cgo CFLAGS: -fno-stack-protector
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#ifndef EAI_NODATA
#define EAI_NODATA -5
#endif

// If nothing else defined EAI_ADDRFAMILY, make sure it has a value.
#ifndef EAI_ADDRFAMILY
#define EAI_ADDRFAMILY -9
#endif

// If nothing else defined EAI_OVERFLOW, make sure it has a value.
#ifndef EAI_OVERFLOW
#define EAI_OVERFLOW -12
#endif
*/
import "C"
import "unsafe"

const (
	_C_AF_INET        = C.AF_INET
	_C_AF_INET6       = C.AF_INET6
	_C_AF_UNSPEC      = C.AF_UNSPEC
	_C_EAI_ADDRFAMILY = C.EAI_ADDRFAMILY
	_C_EAI_AGAIN      = C.EAI_AGAIN
	_C_EAI_NODATA     = C.EAI_NODATA
	_C_EAI_NONAME     = C.EAI_NONAME
	_C_EAI_SERVICE    = C.EAI_SERVICE
	_C_EAI_OVERFLOW   = C.EAI_OVERFLOW
	_C_EAI_SYSTEM     = C.EAI_SYSTEM
	_C_IPPROTO_TCP    = C.IPPROTO_TCP
	_C_IPPROTO_UDP    = C.IPPROTO_UDP
	_C_SOCK_DGRAM     = C.SOCK_DGRAM
	_C_SOCK_STREAM    = C.SOCK_STREAM
)

type (
	_C_char            = C.char
	_C_uchar           = C.uchar
	_C_int             = C.int
	_C_uint            = C.uint
	_C_socklen_t       = C.socklen_t
	_C_struct_addrinfo = C.struct_addrinfo
	_C_struct_sockaddr = C.struct_sockaddr
)

func _C_malloc(n uintptr) unsafe.Pointer { return C.malloc(C.size_t(n)) }
func _C_free(p unsafe.Pointer)           { C.free(p) }

func _C_ai_addr(ai *_C_struct_addrinfo) **_C_struct_sockaddr { return &ai.ai_addr }
func _C_ai_family(ai *_C_struct_addrinfo) *_C_int            { return &ai.ai_family }
func _C_ai_flags(ai *_C_struct_addrinfo) *_C_int             { return &ai.ai_flags }
func _C_ai_next(ai *_C_struct_addrinfo) **_C_struct_addrinfo { return &ai.ai_next }
func _C_ai_protocol(ai *_C_struct_addrinfo) *_C_int          { return &ai.ai_protocol }
func _C_ai_socktype(ai *_C_struct_addrinfo) *_C_int          { return &ai.ai_socktype }

func _C_freeaddrinfo(ai *_C_struct_addrinfo) {
	C.freeaddrinfo(ai)
}

func _C_gai_strerror(eai _C_int) string {
	return C.GoString(C.gai_strerror(eai))
}

func _C_getaddrinfo(hostname, servname *_C_char, hints *_C_struct_addrinfo, res **_C_struct_addrinfo) (int, error) {
	x, err := C.getaddrinfo(hostname, servname, hints, res)
	return int(x), err
}
