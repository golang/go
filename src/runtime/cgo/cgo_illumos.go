// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan

package cgo

// Illumos provides pthread_attr_get_np but not pthread_getattr_np. Importing
// only the function that exists makes an accidental reference to the other
// fail at link time rather than at run time.

//go:cgo_import_dynamic _cgo_libc_pthread_attr_get_np pthread_attr_get_np "libc.so"
