// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	var values []int
	vf(values /* ERROR "(variable of type []int) as string value" */)
	vf(values...) /* ERROR "have (...int)" */
	vf("ab", "cd", values /* ERROR "have (string, string, ...int)" */ ...)
}

func vf(method string, values ...int) {
}
