// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[_ comparable]() {}

type S1 struct{ x int }
type S2 struct{ x any }
type S3 struct{ x [10]interface{ m() } }

func _[P1 comparable, P2 S2]() {
	_ = f[S1]
	_ = f[S2 /* ERROR S2 does not implement comparable */ ]
	_ = f[S3 /* ERROR S3 does not implement comparable */ ]

	type L1 struct { x P1 }
	type L2 struct { x P2 }
	_ = f[L1]
	_ = f[L2 /* ERROR L2 does not implement comparable */ ]
}


// example from issue

type Set[T comparable] map[T]struct{}

func NewSetFromSlice[T comparable](items []T) *Set[T] {
	s := Set[T]{}

	for _, item := range items {
		s[item] = struct{}{}
	}

	return &s
}

type T struct{ x any }

func main() {
	NewSetFromSlice( /* ERROR T does not implement comparable */ []T{
		{"foo"},
		{5},
	})
}
