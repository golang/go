// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue22198

func f(a *bool, b bool) {
	if b {
		return
	}
	c := '\n'
	if b {
		c = ' '
	}
	*a = c == '\n'
}
