// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(p []byte) int {
	switch "" < string(p) {
	case true:
		return 0
	default:
		return 1
	}
}
