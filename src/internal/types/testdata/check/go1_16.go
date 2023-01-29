// -lang=go1.16

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

package p

type Slice []byte
type Array [8]byte

var s Slice
var p = (*Array)(s /* ERROR "requires go1.17 or later" */ )
