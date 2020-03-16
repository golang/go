// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	var B bool
	B2 := (B || B && !B) && !B
	B3 := B2 || B
	for (B3 || B2) && !B2 && B {
	}
}
