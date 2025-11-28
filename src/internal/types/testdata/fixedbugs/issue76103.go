// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	f(foo /* ERROR "undefined: foo" */) // ERROR "not enough arguments in call to f\n\thave (unknown type)\n\twant (int, int)"
}

func f(_, _ int) {}

// test case from issue

type S struct{}

func (S) G() {}

func main() {
	var s S
	_ = must(s.F /* ERROR "s.F undefined" */ ()) // ERROR "not enough arguments in call to must\n\thave (unknown type)\n\twant (T, error)"
}

func must[T any](x T, err error) T {
	if err != nil {
		panic(err)
	}
	return x
}
