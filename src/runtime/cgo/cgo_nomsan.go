// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && (unix || windows) && !(msan && ((linux && (amd64 || arm64 || loong64)) || (freebsd && amd64)))

package cgo

import "unsafe"

//go:nosplit
//go:norace
func msanUnpoison(addr unsafe.Pointer, size uintptr) {}

//go:nosplit
//go:norace
func msanUnpoisonMode(noG bool, addr unsafe.Pointer, size uintptr) {}
