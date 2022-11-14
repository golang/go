// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check error message for use of = instead of == .

package p

func _() {
	if true || 0 /* ERROR cannot use assignment .* as value */ = 1 {
	}
}

func _(a, b string) {
	if a == "a" && b /* ERROR cannot use assignment .* as value */ = "b" {
	}
}
