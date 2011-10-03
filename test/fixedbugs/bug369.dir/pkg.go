// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg

func NonASCII(b []byte, i int) int {
	for i = 0; i < len(b); i++ {
		if b[i] >= 0x80 {
			break
		}
	}
	return i
}

