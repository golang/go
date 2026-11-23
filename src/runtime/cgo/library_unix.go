// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && unix

package cgo

import (
	"errors"
	"internal/stringslite"
	"runtime"
	"sync"
	"unsafe"
)

// dlMu serializes dlopen/dlsym/dlerror calls because POSIX does not guarantee
// that dlerror uses thread-local storage (for example on OpenBSD/Solaris).
var dlMu sync.Mutex

//go:nocheckptr
func cStringToGo(p uintptr) string {
	if p == 0 {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Pointer(p + uintptr(n))) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
}

func dlerrorLocked() string {
	return cStringToGo(asmcgocall(uintptr(libc_dlerror), 0, 0, 0, 0, 0, 0))
}

func dlopen(path string) (uintptr, error) {
	if stringslite.IndexByte(path, 0) >= 0 {
		return 0, errors.New("cgo: dlopen: path contains NUL byte")
	}
	cpath := append([]byte(path), 0)
	dlMu.Lock()
	defer dlMu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tsanAcquire()
	defer tsanRelease()

	asmcgocall(uintptr(libc_dlerror), 0, 0, 0, 0, 0, 0)
	msanUnpoison(unsafe.Pointer(&cpath[0]), uintptr(len(cpath)))
	h := cgocall(uintptr(libc_dlopen), uintptr(unsafe.Pointer(&cpath[0])), dlopenFlags, 0, 0, 0, 0)
	runtime.KeepAlive(cpath)
	if h == 0 {
		msg := dlerrorLocked()
		if msg == "" {
			msg = path + ": unknown dlopen error"
		}
		return 0, errors.New("cgo: dlopen: " + msg)
	}
	return h, nil
}

func dlsym(h uintptr, name string) (unsafe.Pointer, string) {
	if stringslite.IndexByte(name, 0) >= 0 {
		return nil, "symbol name contains NUL byte"
	}
	cname := append([]byte(name), 0)
	dlMu.Lock()
	defer dlMu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tsanAcquire()
	defer tsanRelease()

	asmcgocall(uintptr(libc_dlerror), 0, 0, 0, 0, 0, 0)
	msanUnpoison(unsafe.Pointer(&cname[0]), uintptr(len(cname)))
	ret := cgocall(uintptr(libc_dlsym), h, uintptr(unsafe.Pointer(&cname[0])), 0, 0, 0, 0)
	runtime.KeepAlive(cname)
	if ret == 0 {
		msg := dlerrorLocked()
		if msg == "" {
			msg = "unknown error"
		}
		return nil, msg
	}
	return *(*unsafe.Pointer)(unsafe.Pointer(&ret)), ""
}
