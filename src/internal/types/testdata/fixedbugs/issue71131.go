// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	type Bool bool
	for range func /* ERROR "yield func returns user-defined boolean, not bool" */ (func() Bool) {} {
	}
	for range func /* ERROR "yield func returns user-defined boolean, not bool" */ (func(int) Bool) {} {
	}
	for range func /* ERROR "yield func returns user-defined boolean, not bool" */ (func(int, string) Bool) {} {
	}
}
