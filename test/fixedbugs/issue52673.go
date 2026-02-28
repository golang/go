// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	var x string
	func() [10][]bool {
		return [10][]bool{
			[]bool{bool(x < "")},
			[]bool{}, []bool{}, []bool{}, []bool{}}
	}()
}
