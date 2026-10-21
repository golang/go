// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
)

type Map[K a.Ordered, V any] struct {
	root *node
}

type node struct {
	key any
}

// less orders type-asserted values of the type parameter, the shape of
// gopls internal/util/persistent.
func (*Map[K, V]) less(l, r any) bool {
	return l.(K) < r.(K)
}

func (m *Map[K, V]) before(key K, n *node) bool {
	return key < n.key.(K) && !(n.key.(K) < key)
}

func Max[T a.Ordered](x, y T) T {
	if x < y {
		return y
	}
	return x
}

func Sum[T a.Number](xs ...T) (s T) {
	for _, x := range xs {
		s += x
	}
	return s
}

func Tag[T a.Labeled](x T) string {
	return x.Label()
}

func Wrap[T a.Wrapped](x, y T) string {
	if x < y {
		return y.Wrap()
	}
	return x.Wrap()
}

type label int

func (l label) Label() string { return fmt.Sprint("L", int(l)) }

type word string

func (w word) Wrap() string { return "<" + string(w) + ">" }

func check(ok bool, what string) {
	if !ok {
		panic("go union constraint export: " + what)
	}
}

func main() {
	var m *Map[string, int]
	check(m.less("a", "b") && !m.less("b", "a"), "less")
	check(m.before("a", &node{key: "b"}), "before")
	check(Max(3, 7) == 7, "Max int")
	check(Max("b", "a") == "b", "Max string")
	check(Max(2.5, 1.5) == 2.5, "Max float64")
	check(Max(uint8(9), 4) == 9, "Max uint8")
	check(Sum(1, 2, 3) == 6, "Sum int")
	check(Sum(1.5, 2.5) == 4, "Sum float64")
	check(Tag(label(3)) == "L3", "Tag")
	check(Wrap(word("a"), word("b")) == "<b>", "Wrap")
}
