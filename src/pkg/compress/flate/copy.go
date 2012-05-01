// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// forwardCopy is like the built-in copy function except that it always goes
// forward from the start, even if the dst and src overlap.
func forwardCopy(dst, src []byte) int {
	if len(src) > len(dst) {
		src = src[:len(dst)]
	}
	for i, x := range src {
		dst[i] = x
	}
	return len(src)
}
