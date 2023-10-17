// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure NaN-NaN compiles correctly.

package p

func f() {
	var st struct {
		f    float64
		_, _ string
	}

	f := 1e308
	st.f = 2*f - 2*f
}
