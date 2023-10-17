// compile -d=checkptr

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type ptr unsafe.Pointer

func f(p ptr) *int { return (*int)(p) }
func g(p ptr) ptr  { return ptr(uintptr(p) + 1) }
