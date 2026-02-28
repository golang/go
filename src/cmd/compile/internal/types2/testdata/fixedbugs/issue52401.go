// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	const x = 0
	x /* ERROR cannot assign to x */ += 1
	x /* ERROR cannot assign to x */ ++
}
