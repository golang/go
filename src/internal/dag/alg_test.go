// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dag

import (
	"reflect"
	"strings"
	"testing"
)

func TestTranspose(t *testing.T) {
	g := mustParse(t, diamond)
	g.Transpose()
	wantEdges(t, g, "a->b a->c a->d b->d c->d")
}

func TestTopo(t *testing.T) {
	g := mustParse(t, diamond)
	got := g.Topo()
	// "d" is the root, so it's first.
	//
	// "c" and "b" could be in either order, but Topo is
	// deterministic in reverse node definition order.
	//
	// "a" is a leaf.
	wantNodes := strings.Fields("d c b a")
	if !reflect.DeepEqual(wantNodes, got) {
		t.Fatalf("want topo sort %v, got %v", wantNodes, got)
	}
}

func TestTransitiveReduction(t *testing.T) {
	t.Run("diamond", func { t ->
		g := mustParse(t, diamond)
		g.TransitiveReduction()
		wantEdges(t, g, "b->a c->a d->b d->c")
	})
	t.Run("chain", func { t ->
		const chain = `NONE < a < b < c < d; a, d < e;`
		g := mustParse(t, chain)
		g.TransitiveReduction()
		wantEdges(t, g, "e->d d->c c->b b->a")
	})
}
