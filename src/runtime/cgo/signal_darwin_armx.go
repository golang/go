// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build arm arm64

package cgo

import _ "unsafe"

//go:cgo_export_static xx_cgo_panicmem xx_cgo_panicmem
func xx_cgo_panicmem()
