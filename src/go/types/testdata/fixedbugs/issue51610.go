// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P int | float64 | complex128]() {
	_ = map[P]int{1: 1, 1.0 /* ERROR duplicate key 1 */ : 2, 1 /* ERROR duplicate key \(1 \+ 0i\) */ + 0i: 3}
}
