// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P any](x P) {
	var s []byte
	_ = append(s, x /* ERROR "cannot use x (variable of type P constrained by any) as []byte value in argument to append" */ ...)
	copy(s, x /* ERROR "invalid copy: argument must be a slice; have x (variable of type P constrained by any)" */)
}
