// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g() {}

func f() {
	g()[:] // ERROR "g.. used as value"
}

func g2() ([]byte, []byte) { return nil, nil }

func f2() {
	g2()[:] // ERROR "multiple-value g2.. in single-value context"
}
