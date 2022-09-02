// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type number interface {
	~float64 | ~int | ~int32
	float64 | ~int32
}

func f[T number]() {}

func _() {
	_ = f[int /* ERROR int does not implement number \(int missing in float64 | ~int32\)*/]
}
