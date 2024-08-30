// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "cmd/link/internal/loader"

// Min-heap implementation, for the deadcode pass.
// Specialized for loader.Sym elements.

type heap []loader.Sym

func (h *heap) push(s loader.Sym) {
	*h = append(*h, s)
	// sift up
	n := len(*h) - 1
	for n > 0 {
		p := (n - 1) / 2 // parent
		if (*h)[p] <= (*h)[n] {
			break
		}
		(*h)[n], (*h)[p] = (*h)[p], (*h)[n]
		n = p
	}
}

func (h *heap) pop() loader.Sym {
	r := (*h)[0]
	n := len(*h) - 1
	(*h)[0] = (*h)[n]
	*h = (*h)[:n]

	// sift down
	i := 0
	for {
		c := 2*i + 1 // left child
		if c >= n {
			break
		}
		if c1 := c + 1; c1 < n && (*h)[c1] < (*h)[c] {
			c = c1 // right child
		}
		if (*h)[i] <= (*h)[c] {
			break
		}
		(*h)[i], (*h)[c] = (*h)[c], (*h)[i]
		i = c
	}

	return r
}

func (h *heap) empty() bool { return len(*h) == 0 }

// Same as heap, but sorts alphabetically instead of by index.
// (Note that performance is not so critical here, as it is
// in the case above. Some simplification might be in order.)
type lexHeap []loader.Sym

func (h *lexHeap) push(ldr *loader.Loader, s loader.Sym) {
	*h = append(*h, s)
	// sift up
	n := len(*h) - 1
	for n > 0 {
		p := (n - 1) / 2 // parent
		if ldr.SymName((*h)[p]) <= ldr.SymName((*h)[n]) {
			break
		}
		(*h)[n], (*h)[p] = (*h)[p], (*h)[n]
		n = p
	}
}

func (h *lexHeap) pop(ldr *loader.Loader) loader.Sym {
	r := (*h)[0]
	n := len(*h) - 1
	(*h)[0] = (*h)[n]
	*h = (*h)[:n]

	// sift down
	i := 0
	for {
		c := 2*i + 1 // left child
		if c >= n {
			break
		}
		if c1 := c + 1; c1 < n && ldr.SymName((*h)[c1]) < ldr.SymName((*h)[c]) {
			c = c1 // right child
		}
		if ldr.SymName((*h)[i]) <= ldr.SymName((*h)[c]) {
			break
		}
		(*h)[i], (*h)[c] = (*h)[c], (*h)[i]
		i = c
	}

	return r
}

func (h *lexHeap) empty() bool { return len(*h) == 0 }
