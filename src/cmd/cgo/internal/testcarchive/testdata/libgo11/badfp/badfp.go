// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package badfp is separate from libgo11 because a package that uses cgo
// cannot contain Go assembly.
package badfp

// Fault dereferences nil with a non-canonical frame pointer.
func Fault() // in fault_arm64.s
