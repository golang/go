// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && !msan && !tsan && linux && !android

package cgo

// dlopen flags from <dlfcn.h> in glibc and musl.
const (
	_RTLD_NOW   = 0x2
	_RTLD_LOCAL = 0x0

	dlopenFlags = _RTLD_NOW | _RTLD_LOCAL
)

// Open is the only user of dlopen, dlsym, and dlerror, so link against
// libdl only when the cgobindings experiment is enabled.

//go:cgo_ldflag "-ldl"

//go:cgo_import_dynamic _cgo_libc_dlopen dlopen "libdl.so.2"
//go:cgo_import_dynamic _cgo_libc_dlsym dlsym "libdl.so.2"
//go:cgo_import_dynamic _cgo_libc_dlerror dlerror "libdl.so.2"
