// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func testLCAgen(t *testing.T, bg blockGen, size int) {
	c := testConfig(t)
	fun := c.Fun("entry", bg(size)...)
	CheckFunc(fun.f)
	if size == 4 {
		t.Logf(fun.f.String())
	}
	lca1 := makeLCArange(fun.f)
	lca2 := makeLCAeasy(fun.f)
	for _, b := range fun.f.Blocks {
		for _, c := range fun.f.Blocks {
			l1 := lca1.find(b, c)
			l2 := lca2.find(b, c)
			if l1 != l2 {
				t.Errorf("lca(%s,%s)=%s, want %s", b, c, l1, l2)
			}
		}
	}
}

func TestLCALinear(t *testing.T) {
	testLCAgen(t, genLinear, 10)
	testLCAgen(t, genLinear, 100)
}

func TestLCAFwdBack(t *testing.T) {
	testLCAgen(t, genFwdBack, 10)
	testLCAgen(t, genFwdBack, 100)
}

func TestLCAManyPred(t *testing.T) {
	testLCAgen(t, genManyPred, 10)
	testLCAgen(t, genManyPred, 100)
}

func TestLCAMaxPred(t *testing.T) {
	testLCAgen(t, genMaxPred, 10)
	testLCAgen(t, genMaxPred, 100)
}

func TestLCAMaxPredValue(t *testing.T) {
	testLCAgen(t, genMaxPredValue, 10)
	testLCAgen(t, genMaxPredValue, 100)
}

// Simple implementation of LCA to compare against.
type lcaEasy struct {
	parent []*Block
}

func makeLCAeasy(f *Func) *lcaEasy {
	return &lcaEasy{parent: dominators(f)}
}

func (lca *lcaEasy) find(a, b *Block) *Block {
	da := lca.depth(a)
	db := lca.depth(b)
	for da > db {
		da--
		a = lca.parent[a.ID]
	}
	for da < db {
		db--
		b = lca.parent[b.ID]
	}
	for a != b {
		a = lca.parent[a.ID]
		b = lca.parent[b.ID]
	}
	return a
}

func (lca *lcaEasy) depth(b *Block) int {
	n := 0
	for b != nil {
		b = lca.parent[b.ID]
		n++
	}
	return n
}
