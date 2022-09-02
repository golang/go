// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The core type of M2 unifies with the type of m1
// during function argument type inference.
// M2's constraint is unnamed.
func f1[K1 comparable, E1 any](m1 map[K1]E1) {}

func f2[M2 map[string]int](m2 M2) {
	f1(m2)
}

// The core type of M3 unifies with the type of m1
// during function argument type inference.
// M3's constraint is named.
type Map3 map[string]int

func f3[M3 Map3](m3 M3) {
	f1(m3)
}

// The core type of M5 unifies with the core type of M4
// during constraint type inference.
func f4[M4 map[K4]int, K4 comparable](m4 M4) {}

func f5[M5 map[K5]int, K5 comparable](m5 M5) {
	f4(m5)
}

// test case from issue

func Copy[MC ~map[KC]VC, KC comparable, VC any](dst, src MC) {
	for k, v := range src {
		dst[k] = v
	}
}

func Merge[MM ~map[KM]VM, KM comparable, VM any](ms ...MM) MM {
	result := MM{}
	for _, m := range ms {
		Copy(result, m)
	}
	return result
}
