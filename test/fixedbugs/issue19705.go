// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1() {
	f2()
}

func f2() {
	if false {
		_ = func() {}
	}
}
