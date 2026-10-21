// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// A constraint built from nested unions, none of which the package
// itself instantiates, still supports the ordered operators and
// addition on a type parameter and on type-asserted values of it. This
// is the shape of golang.org/x/tools/gopls/internal/util/constraints
// and persistent.Map.
//
// Checker-level guard only: this single-package check passes with or
// without the exporter fix. The RED regression coverage for the export
// bug lives in test/typeparam/logos_union_constraint_export and
// test/typeparam/logos_union_constraint_misuse.

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type Integer interface {
	Signed | Unsigned
}

type Float interface {
	~float32 | ~float64
}

type Ordered interface {
	Integer | Float | ~string
}

type Map[K Ordered, V any] struct {
	root *mapNode
}

type mapNode struct {
	key any
}

func (*Map[K, V]) less(l, r any) bool {
	return l.(K) < r.(K)
}

func (*Map[K, V]) before(key K, n *mapNode) bool {
	return key < n.key.(K) || n.key.(K) < key
}

func (*Map[K, V]) after(key K, n *mapNode) bool {
	return key >= n.key.(K) && n.key.(K) <= key || n.key.(K) > key
}

func cmp[T Ordered](x, y T) bool {
	return x < y || x <= y || x > y || x >= y
}

func add[T Ordered](x, y T) T {
	return x + y
}

func sub[T Integer](x, y T) T {
	return x - y
}

type S struct{}

var _ = cmp[string]
var _ = cmp[int8]
var _ = cmp[float64]
var _ = cmp[bool /* ERROR "bool does not satisfy Ordered" */]
var _ = cmp[S /* ERROR "S does not satisfy Ordered" */]
var _ = sub[float64 /* ERROR "float64 does not satisfy Integer" */]
