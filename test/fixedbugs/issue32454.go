// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct {
	s string
	f float64
}

func f() {
	var f float64
	var st T
	for {
		switch &st.f {
		case &f:
			f = 1
		}
	}
}
