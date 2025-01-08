// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
M:
L:
	for range 0 {
		break L
		break /* ERROR invalid break label M */ M
	}
	for range 0 {
		break /* ERROR invalid break label L */ L
	}
}
