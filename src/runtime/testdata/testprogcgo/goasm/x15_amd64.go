// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goasm contains Go assembly functions used by testprogcgo because
// packages using cgo can't also contain Go assembly.
package goasm

// ReadX15 returns the lower 64-bits of X15.
func ReadX15() uint64
