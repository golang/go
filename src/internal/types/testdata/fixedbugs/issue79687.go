// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	copy(nil /* ERROR "argument must be a slice; have untyped nil" */, []byte{})
}

// test case from issue

func f() {
	var raw []byte

	copy(nil /* ERROR "argument must be a slice; have untyped nil" */, raw)
}
