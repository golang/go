// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type nat []int

var a, b nat = y()

func y() (nat, []int) {
	return nat{0}, nat{1}
}
