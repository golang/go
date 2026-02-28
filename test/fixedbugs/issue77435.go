// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77435: compiler crash on clear of map resulting
// from a map lookup (or some other syntax that is
// non-idempotent during walk).

package p

func f(s map[int]map[int]int) {
	clear(s[0])
}
