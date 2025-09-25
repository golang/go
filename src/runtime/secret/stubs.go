// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 || amd64

// testing stubs, these are implemented in assembly in
// asm_$GOARCH.s
//
// Note that this file is also used as a template to build a
// crashing binary that tries to leave secrets in places where
// they are supposed to be erased. see crash_test.go for more info

package secret

import "unsafe"

// Load data from p into test registers.
//
//go:noescape
func loadRegisters(p unsafe.Pointer)

// Spill data from test registers into p.
// Returns the amount of space filled in.
//
//go:noescape
func spillRegisters(p unsafe.Pointer) uintptr

// Load secret into all registers.
//
//go:noescape
func useSecret(secret []byte)
