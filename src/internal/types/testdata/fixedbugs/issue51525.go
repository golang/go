// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T interface {
	int
	string
}](x T) {
	_ = x /* ERROR empty type set */ == x
}

func _[T interface{ int | []byte }](x T) {
	_ = x /* ERROR incomparable types in type set */ == x
}
