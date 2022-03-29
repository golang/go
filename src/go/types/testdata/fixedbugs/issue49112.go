// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P int](P) {}

func _() {
        _ = f[int]
        _ = f[[ /* ERROR \[\]int does not implement int */ ]int]

        f(0)
        f/* ERROR \[\]int does not implement int */ ([]int{})
}
