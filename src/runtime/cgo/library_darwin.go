// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && !msan && !tsan && darwin

package cgo

// dlopen flags from <dlfcn.h>. Unlike on most systems, dlopen defaults to
// RTLD_GLOBAL if neither RTLD_LOCAL nor RTLD_GLOBAL is set, so RTLD_LOCAL
// must be passed explicitly.
const (
	_RTLD_NOW   = 0x2
	_RTLD_LOCAL = 0x4

	dlopenFlags = _RTLD_NOW | _RTLD_LOCAL
)

// Open is the only user of dlopen, dlsym, and dlerror, so import them only
// when the cgobindings experiment is enabled.

//go:cgo_import_dynamic libc_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlsym dlsym "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlerror dlerror "/usr/lib/libSystem.B.dylib"
