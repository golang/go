// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(amd64 || arm64) || !linux

package runtime

import "unsafe"

// Stubs for platforms that do not implement runtime/secret

//go:linkname secret_count runtime/secret.count
func secret_count() int32 { return 0 }

//go:linkname secret_inc runtime/secret.inc
func secret_inc() {}

//go:linkname secret_dec runtime/secret.dec
func secret_dec() {}

//go:linkname secret_eraseSecrets runtime/secret.eraseSecrets
func secret_eraseSecrets() {}

func addSecret(p unsafe.Pointer) {}

type specialSecret struct{}

//go:linkname secret_getStack runtime/secret.getStack
func secret_getStack() (uintptr, uintptr) { return 0, 0 }

func noopSignal(mp *m) {}
