// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P ~int | ~complex64](x int, p P) {
	for range real(x /* ERROR "expected complex type" */) { // must not crash
	}
	for range imag(p /* ERROR "expected complex type" */) { // must not crash
	}
	_ = append([]byte{}, complex(x /* ERROR "expected floating-point" */, x)...) // must not crash
}

// test case from issue

func _(x int) {
	for range complex(x /* ERROR "expected floating-point" */, x) {
	}
}
