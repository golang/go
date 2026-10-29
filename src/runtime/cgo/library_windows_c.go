// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && windows && (msan || tsan)

package cgo

/*
#define WIN32_LEAN_AND_MEAN
#include <stdint.h>
#include <windows.h>

static uintptr_t
cgo_load_library(const uint16_t *path, uint32_t *errp) {
	HMODULE h;

	*errp = 0;
	h = LoadLibraryW((LPCWSTR)path);
	if (h == NULL) {
		*errp = (uint32_t)GetLastError();
	}
	return (uintptr_t)h;
}

static uintptr_t
cgo_get_proc_address(uintptr_t h, const char *name, uint32_t *errp) {
	FARPROC p;

	*errp = 0;
	p = GetProcAddress((HMODULE)h, name);
	if (p == NULL) {
		*errp = (uint32_t)GetLastError();
	}
	return (uintptr_t)p;
}
*/
import "C"

import (
	"errors"
	"internal/strconv"
	"internal/stringslite"
	"unsafe"
)

func dlopen(path string) (uintptr, error) {
	if stringslite.IndexByte(path, 0) >= 0 {
		return 0, errors.New("cgo: dlopen: path contains NUL byte")
	}
	wpath := utf16FromString(path)

	var errno C.uint32_t
	ret := uintptr(C.cgo_load_library((*C.uint16_t)(unsafe.Pointer(&wpath[0])), &errno))
	if ret == 0 {
		return 0, errors.New("cgo: dlopen: " + path + ": error " + strconv.FormatUint(uint64(errno), 10))
	}
	return ret, nil
}

func dlsym(h uintptr, name string) (unsafe.Pointer, string) {
	if stringslite.IndexByte(name, 0) >= 0 {
		return nil, "symbol name contains NUL byte"
	}
	cname := append([]byte(name), 0)

	var errno C.uint32_t
	ret := uintptr(C.cgo_get_proc_address(C.uintptr_t(h), (*C.char)(unsafe.Pointer(&cname[0])), &errno))
	if ret == 0 {
		return nil, "error " + strconv.FormatUint(uint64(errno), 10)
	}
	return *(*unsafe.Pointer)(unsafe.Pointer(&ret)), ""
}

func utf16FromString(s string) []uint16 {
	const (
		replacementChar = '\uFFFD'
		maxRune         = '\U0010FFFF'
		surr1           = 0xd800
		surr2           = 0xdc00
		surr3           = 0xe000
		surrSelf        = 0x10000
	)
	buf := make([]uint16, 0, len(s)+1)
	for _, r := range s {
		switch {
		case 0 <= r && r < surr1, surr3 <= r && r < surrSelf:
			buf = append(buf, uint16(r))
		case surrSelf <= r && r <= maxRune:
			r -= surrSelf
			buf = append(buf, uint16(surr1+(r>>10)&0x3ff), uint16(surr2+r&0x3ff))
		default:
			buf = append(buf, replacementChar)
		}
	}
	return append(buf, 0)
}
