// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[T any](T) {}

func _() {
	type F func(int)
	_ = struct{ f F }{f: g}
	_ = [42]F{g}
	_ = []F{g}
	_ = map[int]F{42: g}
	_ = F(g)
	make(chan F) <- g
}

func _[F func(int)]() {
	_ = struct{ f F }{f: g}
	_ = [42]F{g}
	_ = []F{g}
	_ = map[int]F{42: g}
	_ = F(g)
	make(chan F) <- g
}

func _[F func(int) | func(uint)]() {
	_ = struct{ f F }{f: g /* ERROR "cannot use generic function g" */}
	_ = [42]F{g /* ERROR "cannot use generic function g" */}
	_ = []F{g /* ERROR "cannot use generic function g" */}
	_ = map[int]F{42: g /* ERROR "cannot use generic function g" */}
	_ = F(g /* ERROR "cannot use generic function g" */)
	make(chan F) <- g /* ERROR "cannot use generic function g" */
}
