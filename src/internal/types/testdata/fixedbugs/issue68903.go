// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A = [4]int
type B = map[string]interface{}

func _[T ~A](x T) {
	_ = len(x)
}

func _[U ~A](x U) {
	_ = cap(x)
}

func _[V ~A]() {
	_ = V{}
}

func _[W ~B](a interface{}) {
	_ = a.(W)["key"]
}
