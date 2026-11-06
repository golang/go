// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && !msan && !tsan && aix

package cgo

// dlopen flags from <dlfcn.h>. RTLD_MEMBER lets dlopen open archive
// members such as "libc.a(shr_64.o)", the usual form of AIX shared
// libraries.
const (
	_RTLD_NOW    = 0x2
	_RTLD_LOCAL  = 0x80000
	_RTLD_MEMBER = 0x40000

	dlopenFlags = _RTLD_NOW | _RTLD_LOCAL | _RTLD_MEMBER
)

// Open is the only user of dlopen, dlsym, and dlerror, so import them only
// when the cgobindings experiment is enabled.

//go:cgo_import_dynamic _cgo_libc_dlopen dlopen "libc.a/shr_64.o"
//go:cgo_import_dynamic _cgo_libc_dlsym dlsym "libc.a/shr_64.o"
//go:cgo_import_dynamic _cgo_libc_dlerror dlerror "libc.a/shr_64.o"
