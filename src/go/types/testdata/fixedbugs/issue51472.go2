// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T comparable](x T) {
        _ = x == x
}

func _[T interface{interface{comparable}}](x T) {
        _ = x == x
}

func _[T interface{comparable; interface{comparable}}](x T) {
        _ = x == x
}

func _[T interface{comparable; ~int}](x T) {
        _ = x == x
}

func _[T interface{comparable; ~[]byte}](x T) {
        _ = x /* ERROR cannot compare */ == x
}

// TODO(gri) The error message here should be better. See issue #51525.
func _[T interface{comparable; ~int; ~string}](x T) {
        _ = x /* ERROR cannot compare */ == x
}

// TODO(gri) The error message here should be better. See issue #51525.
func _[T interface{~int; ~string}](x T) {
        _ = x /* ERROR cannot compare */ == x
}

func _[T interface{comparable; interface{~int}; interface{int|float64}}](x T) {
        _ = x == x
}

func _[T interface{interface{comparable; ~int}; interface{~float64; comparable; m()}}](x T) {
        _ = x /* ERROR cannot compare */ == x
}

// test case from issue

func f[T interface{comparable; []byte|string}](x T) {
        _ = x == x
}

func _(s []byte) {
	f /* ERROR \[\]byte does not implement interface{comparable; \[\]byte\|string} */ (s)
        _ = f[[ /* ERROR does not implement */ ]byte]
}
