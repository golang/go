// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 78599: compiler ICE (DwarfFixupTable has orphaned fixup)
// when wrapping iter.Seq2[K, ZeroSize] into iter.Seq[K].

package p

import "iter"

func pairs() iter.Seq2[int, struct{}] {
	return func(yield func(int, struct{}) bool) {
		yield(1, struct{}{})
	}
}

func keys() iter.Seq[int] {
	return func(yield func(int) bool) {
		for k := range pairs() {
			if !yield(k) {
				return
			}
		}
	}
}

func use() {
	for range keys() {
	}
}
