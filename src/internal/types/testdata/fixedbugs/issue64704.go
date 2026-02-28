// -lang=go1.21

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	for range 10 /* ERROR "cannot range over 10 (untyped int constant): requires go1.22 or later" */ {
	}
}
