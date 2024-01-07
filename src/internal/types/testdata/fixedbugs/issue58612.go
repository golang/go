// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	var x = new(T)
	f[x /* ERROR "not a type" */ /* ERROR "use of .(type) outside type switch" */ .(type)]()
}

type T struct{}

func f[_ any]() {}
