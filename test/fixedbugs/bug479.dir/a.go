// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type S2 struct {}

const C = unsafe.Sizeof(S2{})

type S1 struct {
	S2
}
