// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

package asan

import (
	"unsafe"
)

const Enabled = true

//go:linkname Read runtime.asanread
func Read(addr unsafe.Pointer, len int)

//go:linkname Write runtime.asanwrite
func Write(addr unsafe.Pointer, len int)
