// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import "unsafe"

//go:linkname _cgo_is_musl _cgo_is_musl
var _cgo_is_musl unsafe.Pointer
