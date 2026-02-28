// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type policy[K, V any] interface{}
type LRU[K, V any] struct{}

func NewCache[K, V any](p policy[K, V]) {}

func _() {
	var lru LRU[int, string]
	NewCache[int, string](&lru)
	NewCache(& /* ERROR does not match policy\[K, V\] \(cannot infer K and V\) */ lru)
}
