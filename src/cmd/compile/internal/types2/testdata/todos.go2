// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is meant as "dumping ground" for tests
// of not yet implemented features. It will grow and
// shrink over time.

package p

// When using []'s instead of ()'s for type parameters
// we don't need extra parentheses for some composite
// literal types.
type T1[P any] struct{}
type T2[P, Q any] struct{}

func _() {
   _ = []T1[int]{}            // ok if we use []'s
   _ = [](T1[int]){}
   _ = []T2[int, string]{}    // ok if we use []'s
   _ = [](T2[int, string]){}
}
