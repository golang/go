// errorcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ chan [0x2FFFF]byte         // ERROR "channel element type too large"
var _ = make(chan [0x2FFFF]byte) // ERROR "channel element type too large"

var c1 chan [0x2FFFF]byte         // ERROR "channel element type too large"
var c2 = make(chan [0x2FFFF]byte) // ERROR "channel element type too large"
