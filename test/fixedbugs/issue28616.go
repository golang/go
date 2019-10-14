// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't dead code eliminate a label.

package p

var i int

func f() {

	if true {

		if i == 1 {
			goto label
		}

		return
	}

label:
}
