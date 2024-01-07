// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type E [1 << 32]byte

var a [1 << 32]E // size of a must not overflow to 0
var _ = unsafe.Sizeof(a /* ERROR "too large" */ )
