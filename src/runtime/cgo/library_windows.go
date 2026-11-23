// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && windows

package cgo

import (
	"errors"
	"internal/strconv"
	"internal/stringslite"
	"runtime"
	"unsafe"
)

//go:cgo_import_dynamic _cgo_LoadLibraryW LoadLibraryW%1 "kernel32.dll"
//go:linkname _cgo_LoadLibraryW _cgo_LoadLibraryW
var _cgo_LoadLibraryW unsafe.Pointer

//go:cgo_import_dynamic _cgo_GetProcAddress GetProcAddress%2 "kernel32.dll"
//go:linkname _cgo_GetProcAddress _cgo_GetProcAddress
var _cgo_GetProcAddress unsafe.Pointer

func dlopen(path string) (uintptr, error) {
	if stringslite.IndexByte(path, 0) >= 0 {
		return 0, errors.New("cgo: dlopen: path contains NUL byte")
	}
	wpath := utf16FromString(path)

	ret, errno := cgocallErr(uintptr(_cgo_LoadLibraryW), uintptr(unsafe.Pointer(&wpath[0])), 0, 0, 0, 0, 0)
	runtime.KeepAlive(wpath)
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

	ret, errno := cgocallErr(uintptr(_cgo_GetProcAddress), h, uintptr(unsafe.Pointer(&cname[0])), 0, 0, 0, 0)
	runtime.KeepAlive(cname)
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
