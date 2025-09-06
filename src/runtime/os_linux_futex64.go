// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// 64-bit systems only. For 32-bit systems, os_linux_futex32.go has a fallback.

//go:build linux && !(386 || arm || mips || mipsle || ppc || s390)

package runtime

import (
	"unsafe"
)

//go:noescape
func futex(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32
