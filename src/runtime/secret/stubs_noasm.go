// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 && !amd64

package secret

import "unsafe"

func loadRegisters(p unsafe.Pointer)          {}
func spillRegisters(p unsafe.Pointer) uintptr { return 0 }
func useSecret(secret []byte)                 {}
