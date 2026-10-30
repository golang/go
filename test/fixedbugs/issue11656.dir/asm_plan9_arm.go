// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"syscall"
)

func syncIcache(p uintptr) {
	r, _, err := syscall.Syscall(syscall.SYS_SEGFLUSH, p, 8, 0)
	if int32(r) == -1 {
		log.Fatalf("segflush: %v", err)
	}
}
