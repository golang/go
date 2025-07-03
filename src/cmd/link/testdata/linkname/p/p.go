// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import _ "unsafe"

// f1 is pushed from main.
//
//go:linkname f1
func f1()

// Push f2 to main.
//
//go:linkname f2 main.f2
func f2() {}

func F() { f1() }
