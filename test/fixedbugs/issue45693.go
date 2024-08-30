// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 45693: ICE with register args.

package p

func f() {
	var s string
	s = s + "" + s + "" + s + ""
	for {
	}
}
