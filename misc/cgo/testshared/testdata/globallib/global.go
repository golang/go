// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package globallib

// Data is large enough to that offsets into it do not fit into
// 16-bit or 20-bit immediates. Ideally we'd also try and overrun
// 32-bit immediates, but that requires the test machine to have
// too much memory.
var Data [1<<20 + 10]int64

func init() {
	for i := range Data {
		Data[i] = int64(i)
	}
}
