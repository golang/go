// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Failed to compile with gccgo.

package p

import "unsafe"

const w int = int(unsafe.Sizeof(0))

var a [w]byte
