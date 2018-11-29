// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 22076: Couldn't use ":=" to declare names that refer to
// dot-imported symbols.

package p

import . "bytes"

var _ Reader // use "bytes" import

func _() {
	Buffer := 0
	_ = Buffer
}

func _() {
	for Buffer := range []int{} {
		_ = Buffer
	}
}
