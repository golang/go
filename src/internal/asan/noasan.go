// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !asan

package asan

import (
	"unsafe"
)

const Enabled = false

func Read(addr unsafe.Pointer, len uintptr) {}

func Write(addr unsafe.Pointer, len uintptr) {}
