// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var bar = f(13579)

func f(x uint16) uint16 {
	return x>>8 | x<<8
}
