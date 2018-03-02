// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func bitcheck(a, b uint64) int {
	if a&(1<<(b&63)) != 0 { // amd64:"BTQ"
		return 1
	}
	return -1
}
