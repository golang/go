// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm32

package goarch

const (
	_ArchFamily          = WASM
	_DefaultPhysPageSize = 65536
	_PCQuantum           = 1
	_MinFrameSize        = 0
	// WASI requires 8-byte alignment for uint64 values on the Go stack.
	// The return-PC slot is also 8 bytes wide.
	_StackAlign = 8
)
