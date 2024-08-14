// -lang=go1.21

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

//go:build go1.19

package p

type Slice []byte
type Array [8]byte

var s Slice
var p = (Array)(s /* ERROR "requires go1.20 or later" */)
