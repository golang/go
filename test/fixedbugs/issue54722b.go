// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type value[V comparable] struct {
	node  *node[value[V]]
	value V
}

type node[V comparable] struct {
	index    *index[V]
	children map[string]*node[V]
}

type index[V comparable] struct {
	arrays []array[V]
}

type array[V comparable] struct {
	valueMap map[int]V
}

var x value[int]
var y value[*Column]

type Column struct{ column int }
