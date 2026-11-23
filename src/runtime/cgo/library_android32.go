// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && android && (386 || arm)

package cgo

// dlopen flags from bionic's <dlfcn.h> for LP32, where RTLD_NOW is 0.
// The runtime/cgo initialization code on Android already imports dlopen,
// dlsym, and dlerror (see cgo_android.go).
const (
	_RTLD_NOW   = 0x0
	_RTLD_LOCAL = 0x0

	dlopenFlags = _RTLD_NOW | _RTLD_LOCAL
)
