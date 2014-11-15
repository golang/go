// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:cgo_export_static main

// Filled in by runtime/cgo when linked into binary.

//go:linkname _cgo_init _cgo_init
//go:linkname _cgo_malloc _cgo_malloc
//go:linkname _cgo_free _cgo_free
//go:linkname _cgo_thread_start _cgo_thread_start

var (
	_cgo_init         unsafe.Pointer
	_cgo_malloc       unsafe.Pointer
	_cgo_free         unsafe.Pointer
	_cgo_thread_start unsafe.Pointer
)
