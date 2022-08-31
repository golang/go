// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Map map[string]int

func f[M ~map[K]V, K comparable, V any](M) {}
func g[M map[K]V, K comparable, V any](M) {}

func _[M1 ~map[K]V, M2 map[K]V, K comparable, V any]() {
        var m1 M1
        f(m1)
        g /* ERROR M1 does not implement map\[K\]V */ (m1) // M1 has tilde

        var m2 M2
        f(m2)
        g(m2) // M1 does not have tilde

        var m3 Map
        f(m3)
        g /* ERROR Map does not implement map\[string\]int */ (m3) // M in g does not have tilde
}
