// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	var x interface{}
	switch t := x.(type) {
	case S /* ERROR "cannot use generic type" */ :
		t.m()
	}
}

type S[T any] struct {}

func (_ S[T]) m()
