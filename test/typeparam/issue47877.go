// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Map[K comparable, V any] struct {
        m map[K]V
}

func NewMap[K comparable, V any]() Map[K, V] {
        return Map[K, V]{m: map[K]V{}}
}

func (m Map[K, V]) Get(key K) V {
        return m.m[key]
}

func main() {
        _ = NewMap[int, struct{}]()
}
