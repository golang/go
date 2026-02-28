// build

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"iter"
)

type leafSet map[rune]struct{}

type branchMap map[rune]*node

func (bm branchMap) findOrCreateBranch(r rune) *node {
	if _, ok := bm[r]; !ok {
		bm[r] = newNode()
	}
	return bm[r]
}

func (bm branchMap) allSuffixes() iter.Seq[string] {
	return func(yield func(string) bool) {
		for r, n := range bm {
			for s := range n.allStrings() {
				if !yield(string(r) + s) {
					return
				}
			}
		}
	}
}

type node struct {
	leafSet
	branchMap
}

func newNode() *node {
	return &node{make(leafSet), make(branchMap)}
}

func (n *node) add(s []rune) {
	switch len(s) {
	case 0:
		return
	case 1:
		n.leafSet[s[0]] = struct{}{}
	default:
		n.branchMap.findOrCreateBranch(s[0]).add(s[1:])
	}
}

func (n *node) addString(s string) {
	n.add([]rune(s))
}

func (n *node) allStrings() iter.Seq[string] {
	return func(yield func(string) bool) {
		for s := range n.leafSet {
			if !yield(string(s)) {
				return
			}
		}
		for r, n := range n.branchMap {
			for s := range n.allSuffixes() {
				if !yield(string(r) + s) {
					return
				}
			}
		}
	}
}

func main() {
	root := newNode()
	for _, s := range []string{"foo", "bar", "baz", "a", "b", "c", "hello", "world"} {
		root.addString(s)
	}
	for s := range root.allStrings() {
		println(s)
	}
}
