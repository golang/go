// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func e() {
	for true {
		if true {
			continue
		}
	}
}

func g() {}

func f() {
	i := 0
	if true {
		i++
	}
	for true {
		continue
		g()
	}
}
