// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(x, y, z int) {
	if x /* ERROR cannot use assignment x = y as value */ = y {}
	if x || y /* ERROR cannot use assignment \(x || y\) = z as value */ = z {}
	if x /* ERROR cannot use assignment x = \(y || z\) as value */ = y || z {}
}
