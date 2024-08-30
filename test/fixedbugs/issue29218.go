// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct {
	b bool
	string
}

func f() {
	var b bool
	var t T
	for {
		switch &t.b {
		case &b:
			if b {
			}
		}
	}
}
