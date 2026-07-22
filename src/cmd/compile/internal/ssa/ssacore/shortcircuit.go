// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

// ReplaceUses replaces all uses of old in b with new.
func (b *Block) ReplaceUses(old, new *Value) {
	for _, v := range b.Values {
		for i, a := range v.Args {
			if a == old {
				v.SetArg(i, new)
			}
		}
	}
	for i, v := range b.ControlValues() {
		if v == old {
			b.ReplaceControl(i, new)
		}
	}
}

// MoveTo moves v to dst, adjusting the appropriate Block.Values slices.
// The caller is responsible for ensuring that this is safe.
// i is the index of v in v.Block.Values.
func (v *Value) MoveTo(dst *Block, i int) {
	if dst.Func.Scheduled {
		v.Fatalf("moveTo after scheduling")
	}
	src := v.Block
	if src.Values[i] != v {
		v.Fatalf("moveTo bad index %d", v, i)
	}
	if src == dst {
		return
	}
	v.Block = dst
	dst.Values = append(dst.Values, v)
	last := len(src.Values) - 1
	src.Values[i] = src.Values[last]
	src.Values[last] = nil
	src.Values = src.Values[:last]
}

// FlowsTo checks that the subgraph starting from v and ends at t is a DAG, with
// the following constraints:
//
//	(1) v can reach t.
//	(2) v's connected component removing the paths containing t is a DAG.
//	(3) The blocks in the subgraph G defined in (2) has all their preds also in G,
//	    except v.
//	(4) The subgraph defined in (2) has a size smaller than cap.
//
//	We know that the subgraph G defined in constraint (2)(3) has the property that v
//	dominates all the blocks in G:
//		If there exist a block x in G that is not dominated by v, then there exist a
//		path P from entry to x that does not contain v. Denote x's predecessor in P
//		as x', then x' must also be in G given constraint (3), same to its pred x''
//		in P. Given constraint (2), by going back in P we will in the end reach v,
//		which conflicts with the definition of P.
//
// Constraint (2)'s DAG requirement could be further relaxed to contain "internal"
// loops that doesn't change the dominance relation of v. But that is more subtle
// and requires another constraint on the source block v, and a more complex proof.
// Furthermore optimizing the branch guarding a loop might bring less gains as the
// loop itself might be the bottleneck.
func (v *Block) FlowsTo(t *Block, cap int) map[*Block]struct{} {
	seen := map[*Block]struct{}{}
	var boundedDFS func(b *Block)
	hasPathToT := false
	fullyExplored := true
	isDAG := true
	visited := map[*Block]struct{}{}
	boundedDFS = func(b *Block) {
		if _, ok := seen[b]; ok {
			return
		}
		if _, ok := visited[b]; ok {
			isDAG = false
			return
		}
		if b == t {
			// do not put t into seen, this way
			// if v can reach t's connected component without going through t,
			// it will fail the pred check after boundedDFSUntil.
			hasPathToT = true
			return
		}
		if len(seen) > cap {
			fullyExplored = false
			return
		}
		seen[b] = struct{}{}
		visited[b] = struct{}{}
		for _, se := range b.Succs {
			boundedDFS(se.B)
			if !(isDAG && fullyExplored) {
				return
			}
		}
		delete(visited, b)
	}
	boundedDFS(v)
	if hasPathToT && fullyExplored && isDAG {
		for b := range seen {
			if b != v {
				for _, se := range b.Preds {
					if _, ok := seen[se.B]; !ok {
						return nil
					}
				}
			}
		}
		return seen
	}
	return nil
}
