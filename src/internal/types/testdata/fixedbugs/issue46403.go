// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue46403

func _() {
	// a should be used, despite the parser error below.
	var a []int
	var _ = a[] // ERROR "expected operand"
}
