// -lang=go1.22

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

//go:build go1.21

package p

func f() {
	for _ = range 10 /* ERROR "requires go1.22 or later" */ {
	}
}