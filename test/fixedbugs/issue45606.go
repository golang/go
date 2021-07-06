// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func x() {
	func() func() {
		return func() {
			f := func() {}
			g, _ := f, 0
			g()
		}
	}()()
}
