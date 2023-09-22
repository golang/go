// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Cache[K comparable, V any] interface{}

type LRU[K comparable, V any] struct{}

func WithLocking2[K comparable, V any](Cache[K, V]) {}

func _() {
	WithLocking2 /* ERROR "cannot infer V" */ [string](LRU[string, int]{})
}
