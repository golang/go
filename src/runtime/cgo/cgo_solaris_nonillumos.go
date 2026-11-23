// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && solaris && !illumos

package cgo

// Solaris provides pthread_getattr_np but not pthread_attr_get_np. Importing
// only the function that exists makes an accidental reference to the other
// fail at link time rather than at run time.

//go:cgo_import_dynamic _cgo_libc_pthread_getattr_np pthread_getattr_np "libc.so"
