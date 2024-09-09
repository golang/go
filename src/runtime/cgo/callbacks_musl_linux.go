// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package cgo

import _ "unsafe" // for go:linkname

// x_cgo_is_musl is set to 1 if the C library is musl.

//go:cgo_import_static x_cgo_is_musl
//go:linkname x_cgo_is_musl x_cgo_is_musl
//go:linkname _cgo_is_musl _cgo_is_musl
var x_cgo_is_musl byte
var _cgo_is_musl = &x_cgo_is_musl
