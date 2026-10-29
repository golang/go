// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && unix && (msan || tsan)

package cgo

/*
#cgo linux LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdint.h>

static uintptr_t
cgo_dlopen(const char *path, char **errp) {
	int flags = RTLD_NOW | RTLD_LOCAL;
	void *h;

#ifdef _AIX
	flags |= RTLD_MEMBER;
#endif
	*errp = NULL;
	dlerror();
	h = dlopen(path, flags);
	if (h == NULL) {
		*errp = (char *)dlerror();
	}
	return (uintptr_t)h;
}

static void *
cgo_dlsym(uintptr_t h, const char *name, char **errp) {
	void *p;

	*errp = NULL;
	dlerror();
	p = dlsym((void *)h, name);
	if (p == NULL) {
		*errp = (char *)dlerror();
	}
	return p;
}
*/
import "C"

import (
	"errors"
	"internal/stringslite"
	"sync"
	"unsafe"
)

// dlMu serializes dlopen/dlsym/dlerror calls because POSIX does not guarantee
// that dlerror uses thread-local storage (for example on OpenBSD/Solaris).
var dlMu sync.Mutex

func dlopen(path string) (uintptr, error) {
	if stringslite.IndexByte(path, 0) >= 0 {
		return 0, errors.New("cgo: dlopen: path contains NUL byte")
	}
	cpath := append([]byte(path), 0)
	dlMu.Lock()
	defer dlMu.Unlock()

	var cErr *C.char
	h := uintptr(C.cgo_dlopen((*C.char)(unsafe.Pointer(&cpath[0])), &cErr))
	if h == 0 {
		msg := ""
		if cErr != nil {
			msg = C.GoString(cErr)
		}
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

	var cErr *C.char
	ret := C.cgo_dlsym(C.uintptr_t(h), (*C.char)(unsafe.Pointer(&cname[0])), &cErr)
	if ret == nil {
		msg := ""
		if cErr != nil {
			msg = C.GoString(cErr)
		}
		if msg == "" {
			msg = "unknown error"
		}
		return nil, msg
	}
	return ret, ""
}
