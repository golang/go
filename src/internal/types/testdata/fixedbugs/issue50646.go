// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1[_ comparable]()              {}
func f2[_ interface{ comparable }]() {}

type T interface{ m() }

func _[P comparable, Q ~int, R any]() {
	_ = f1[int]
	_ = f1[T /* ERROR T does not implement comparable */ ]
	_ = f1[any /* ERROR any does not implement comparable */ ]
	_ = f1[P]
	_ = f1[Q]
	_ = f1[R /* ERROR R does not implement comparable */]

	_ = f2[int]
	_ = f2[T /* ERROR T does not implement comparable */ ]
	_ = f2[any /* ERROR any does not implement comparable */ ]
	_ = f2[P]
	_ = f2[Q]
	_ = f2[R /* ERROR R does not implement comparable */]
}
