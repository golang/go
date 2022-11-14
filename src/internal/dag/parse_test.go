// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dag

import (
	"reflect"
	"strings"
	"testing"
)

const diamond = `
NONE < a < b, c < d;
`

func mustParse(t *testing.T, dag string) *Graph {
	t.Helper()
	g, err := Parse(dag)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func wantEdges(t *testing.T, g *Graph, edges string) {
	t.Helper()

	wantEdges := strings.Fields(edges)
	wantEdgeMap := make(map[string]bool)
	for _, e := range wantEdges {
		wantEdgeMap[e] = true
	}

	for _, n1 := range g.Nodes {
		for _, n2 := range g.Nodes {
			got := g.HasEdge(n1, n2)
			want := wantEdgeMap[n1+"->"+n2]
			if got && want {
				t.Logf("%s->%s", n1, n2)
			} else if got && !want {
				t.Errorf("%s->%s present but not expected", n1, n2)
			} else if want && !got {
				t.Errorf("%s->%s missing but expected", n1, n2)
			}
		}
	}
}

func TestParse(t *testing.T) {
	// Basic smoke test for graph parsing.
	g := mustParse(t, diamond)

	wantNodes := strings.Fields("a b c d")
	if !reflect.DeepEqual(wantNodes, g.Nodes) {
		t.Fatalf("want nodes %v, got %v", wantNodes, g.Nodes)
	}

	// Parse returns the transitive closure, so it adds d->a.
	wantEdges(t, g, "b->a c->a d->a d->b d->c")
}
