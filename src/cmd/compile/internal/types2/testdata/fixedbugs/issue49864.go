// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P ~int, Q any](p P) {
	_ = Q(p /* ERROR cannot convert */ )
}
