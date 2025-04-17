// build

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F(a, b map[float32]int) int {
	var st *struct {
		n int
		f float32
	}
	return a[0] + b[st.f]
}
