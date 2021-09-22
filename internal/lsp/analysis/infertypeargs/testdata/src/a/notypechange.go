// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We should not suggest removing type arguments if doing so would change the
// resulting type.

package a

func id[T any](t T) T { return t }

var _ = id[int](1)        // want "unnecessary type arguments"
var _ = id[string]("foo") // want "unnecessary type arguments"
var _ = id[int64](2)

func pair[T any](t T) (T, T) { return t, t }

var _, _ = pair[int](3) // want "unnecessary type arguments"
var _, _ = pair[int64](3)

func noreturn[T any](t T) {}

func _() {
	noreturn[int64](4)
	noreturn[int](4) // want "unnecessary type arguments"
}
