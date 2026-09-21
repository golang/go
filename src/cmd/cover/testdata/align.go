// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var badSlice [8265]byte

func init() {
	badSlice[0] = 4
}
