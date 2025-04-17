// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	var st struct {
		s string
		i int16
	}
	_ = func() {
		var m map[int16]int
		m[st.i] = 0
	}
}
