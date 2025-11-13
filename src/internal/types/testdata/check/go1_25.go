// -lang=go1.25

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

//go:build go1.25

package p

func f(x int) {
	_ = new /* ERROR "new(123) requires go1.26 or later" */ (123)
	_ = new /* ERROR "new(x) requires go1.26 or later" */ (x)
	_ = new /* ERROR "new(f) requires go1.26 or later" */ (f)
	_ = new /* ERROR "new(1 < 2) requires go1.26 or later" */ (1 < 2)
}
