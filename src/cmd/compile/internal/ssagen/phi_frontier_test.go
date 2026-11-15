// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"testing"
)

func TestPhiFrontierOracle(t *testing.T) {
	tests := []phiFrontierCase{
		{"soleNonEntryLoop", 5, [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 1}, {1, 4}}, []int{2}, nil},
		{"emptyNonEntryFrontier", 4, [][2]int{{0, 1}, {1, 2}, {2, 3}}, []int{1}, nil},
		{"diamond", 5, [][2]int{{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}}, []int{1, 2}, nil},
		{"nestedLoops", 7, [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 2}, {2, 4}, {4, 5}, {5, 1}, {1, 6}}, []int{3, 4}, nil},
		{"rootChildBoundary", 6, [][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 1}, {2, 4}, {4, 5}}, []int{1}, nil},
		{"duplicateEdges", 5, [][2]int{{0, 1}, {0, 2}, {1, 3}, {1, 3}, {2, 3}, {3, 4}}, []int{1, 2}, nil},
		{"definitionAtJoin", 5, [][2]int{{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 1}, {3, 4}}, []int{1, 3}, nil},
		{"unreachableWithoutDefinition", 6, [][2]int{{0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {5, 4}}, []int{1, 2}, nil},
		{"unreachableDefinition", 5, [][2]int{{0, 1}, {1, 2}, {3, 3}}, []int{1, 3, 4}, []int{3}},
	}
	for _, tc := range tests {
		for variant := range 3 {
			t.Run(fmt.Sprintf("%s/order%d", tc.name, variant), func(t *testing.T) {
				tc.exercise(t, variant)
			})
		}
	}
}

func phiFrontierReference(blocks []*ssa.Block, definitions []int) ([][]bool, []bool, []bool) {
	n := len(blocks)
	labels := make(map[*ssa.Block]int)
	for i, b := range blocks {
		labels[b] = i
	}
	reachable := make([]bool, n)
	work := []int{0}
	for len(work) != 0 {
		i := work[len(work)-1]
		work = work[:len(work)-1]
		if reachable[i] {
			continue
		}
		reachable[i] = true
		for _, e := range blocks[i].Succs {
			work = append(work, labels[e.Block()])
		}
	}
	dom := make([][]bool, n)
	for i := range dom {
		dom[i] = make([]bool, n)
		if !reachable[i] {
			continue
		}
		for j := range dom[i] {
			dom[i][j] = reachable[j] && (i != 0 || j == 0)
		}
	}
	for changed := true; changed; {
		changed = false
		for i := 1; i < n; i++ {
			if !reachable[i] {
				continue
			}
			for d := range n {
				value := d == i || (phiDominance{labels, reachable, dom}).allPredecessors(blocks[i], d)
				if dom[i][d] != value {
					dom[i][d] = value
					changed = true
				}
			}
		}
	}
	frontier := make([][]bool, n)
	for x := range blocks {
		frontier[x] = make([]bool, n)
		if !reachable[x] {
			continue
		}
		for y, b := range blocks {
			if !reachable[y] || (x != y && dom[y][x]) {
				continue
			}
			for _, e := range b.Preds {
				pred := labels[e.Block()]
				if reachable[pred] && dom[pred][x] {
					frontier[x][y] = true
				}
			}
		}
	}
	active := make([]bool, n)
	for _, d := range definitions {
		active[d] = reachable[d]
	}
	phis := make([]bool, n)
	for changed := true; changed; {
		changed = false
		for x := range blocks {
			if !active[x] {
				continue
			}
			for y := range blocks {
				if !frontier[x][y] || phis[y] {
					continue
				}
				phis[y] = true
				active[y] = true
				changed = true
			}
		}
	}
	return dom, reachable, phis
}

type phiFrontierCase struct {
	name     string
	n        int
	edges    [][2]int
	defs     []int
	deadPhis []int
}

func (tc phiFrontierCase) exercise(t *testing.T, variant int) {
	f := &ssa.Func{Cache: new(ssa.Cache)}
	blocks := make([]*ssa.Block, tc.n)
	for i := range tc.n {
		label := i
		if variant == 1 {
			label = tc.n - 1 - i
		}
		blocks[label] = f.NewBlock(ssa.BlockPlain)
	}
	f.Entry = blocks[0]
	for i := range tc.edges {
		j := i
		if variant == 2 {
			j = len(tc.edges) - 1 - i
		}
		e := tc.edges[j]
		blocks[e[0]].AddEdgeTo(blocks[e[1]])
	}
	dom, reachable, want := phiFrontierReference(blocks, tc.defs)
	for _, b := range tc.deadPhis {
		want[b] = true
	}
	variable := ir.NewNilExpr(src.NoXPos, types.TypeInvalid)
	p := &phiState{
		f: f, s: &state{blockStarts: make([]src.XPos, f.NumBlocks())},
		idom: f.Idom(), tree: make([]domBlock, f.NumBlocks()),
		level:   make([]int32, f.NumBlocks()),
		defvars: make([]map[ir.Node]*ssa.Value, f.NumBlocks()),
		varnum:  map[ir.Node]int32{variable: 0},
		queued:  newSparseSet(f.NumBlocks()), hasPhi: newSparseSet(f.NumBlocks()), hasDef: newSparseSet(f.NumBlocks()),
	}
	for _, b := range blocks {
		if parent := p.idom[b.ID]; parent != nil {
			p.tree[b.ID].sibling = p.tree[parent.ID].firstChild
			p.tree[parent.ID].firstChild = b
		}
		for a := p.idom[b.ID]; a != nil; a = p.idom[a.ID] {
			p.level[b.ID]++
		}
	}
	p.priq.level = p.level
	p.placeholder = f.Entry.NewValue0(src.NoXPos, ssa.OpUnknown, types.TypeInvalid)
	defValues := make(map[int]*ssa.Value)
	var definitions []*ssa.Block
	for i := range tc.defs {
		j := i
		if variant != 0 {
			j = len(tc.defs) - 1 - i
		}
		label := tc.defs[j]
		b := blocks[label]
		v := b.NewValue0I(src.NoXPos, ssa.OpConst64, types.TypeInvalid, int64(label+1))
		defValues[label] = v
		p.defvars[b.ID] = map[ir.Node]*ssa.Value{variable: v}
		definitions = append(definitions, b)
	}
	p.computeMinSuccessorLevels()
	p.insertVarPhis(0, variable, definitions, types.TypeInvalid)
	phis := make(map[int]*ssa.Value)
	for label, b := range blocks {
		for _, v := range b.Values {
			if v.Op != ssa.OpPhi {
				continue
			}
			if phis[label] != nil {
				t.Fatalf("duplicate phi in block %d", label)
			}
			phis[label] = v
			if len(v.Args) != len(b.Preds) {
				t.Fatalf("block %d: %d args for %d predecessors", label, len(v.Args), len(b.Preds))
			}
		}
		if (phis[label] != nil) != want[label] {
			t.Fatalf("block %d: phi=%v, oracle=%v", label, phis[label] != nil, want[label])
		}
	}
	p.resolveFwdRefs()
	labels := make(map[*ssa.Block]int)
	for label, b := range blocks {
		labels[b] = label
	}
	for label, phi := range phis {
		for i, edge := range blocks[label].Preds {
			pred := labels[edge.Block()]
			expected := (phiReachingValues{dom, reachable, defValues, phis, p.placeholder}).outgoing(pred)
			if phi.Args[i] != expected {
				t.Errorf("block %d predecessor slot %d (block %d): got %v, want %v", label, i, pred, phi.Args[i], expected)
			}
		}
	}
}

type phiDominance struct {
	labels    map[*ssa.Block]int
	reachable []bool
	dom       [][]bool
}

func (r phiDominance) allPredecessors(b *ssa.Block, d int) bool {
	if !r.reachable[d] {
		return false
	}
	for _, e := range b.Preds {
		pred := r.labels[e.Block()]
		if r.reachable[pred] && !r.dom[pred][d] {
			return false
		}
	}
	return true
}

type phiReachingValues struct {
	dom         [][]bool
	reachable   []bool
	definitions map[int]*ssa.Value
	phis        map[int]*ssa.Value
	unknown     *ssa.Value
}

func (r phiReachingValues) outgoing(pred int) *ssa.Value {
	if !r.reachable[pred] {
		return r.unknown
	}
	best := -1
	for d := range r.dom[pred] {
		if !r.dom[pred][d] || (r.definitions[d] == nil && r.phis[d] == nil) {
			continue
		}
		if best == -1 || r.dom[d][best] {
			best = d
		}
	}
	if best == -1 {
		return r.unknown
	}
	if v := r.definitions[best]; v != nil {
		return v
	}
	return r.phis[best]
}
