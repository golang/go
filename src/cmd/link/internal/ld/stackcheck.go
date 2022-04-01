// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"fmt"
	"internal/buildcfg"
	"sort"
	"strings"
)

type stackCheck struct {
	ctxt      *Link
	ldr       *loader.Loader
	morestack loader.Sym
	callSize  int // The number of bytes added by a CALL

	// height records the maximum number of bytes a function and
	// its callees can add to the stack without a split check.
	height map[loader.Sym]int16

	// graph records the out-edges from each symbol. This is only
	// populated on a second pass if the first pass reveals an
	// over-limit function.
	graph map[loader.Sym][]stackCheckEdge
}

type stackCheckEdge struct {
	growth int        // Stack growth in bytes at call to target
	target loader.Sym // 0 for stack growth without a call
}

// stackCheckCycle is a sentinel stored in the height map to detect if
// we've found a cycle. This is effectively an "infinite" stack
// height, so we use the closest value to infinity that we can.
const stackCheckCycle int16 = 1<<15 - 1

// stackCheckIndirect is a sentinel Sym value used to represent the
// target of an indirect/closure call.
const stackCheckIndirect loader.Sym = -1

// doStackCheck walks the call tree to check that there is always
// enough stack space for call frames, especially for a chain of
// nosplit functions.
//
// It walks all functions to accumulate the number of bytes they can
// grow the stack by without a split check and checks this against the
// limit.
func (ctxt *Link) doStackCheck() {
	sc := newStackCheck(ctxt, false)

	// limit is number of bytes a splittable function ensures are
	// available on the stack. If any call chain exceeds this
	// depth, the stack check test fails.
	//
	// The call to morestack in every splittable function ensures
	// that there are at least StackLimit bytes available below SP
	// when morestack returns.
	limit := objabi.StackLimit - sc.callSize
	if buildcfg.GOARCH == "arm64" {
		// Need an extra 8 bytes below SP to save FP.
		limit -= 8
	}

	// Compute stack heights without any back-tracking information.
	// This will almost certainly succeed and we can simply
	// return. If it fails, we do a second pass with back-tracking
	// to produce a good error message.
	//
	// This accumulates stack heights bottom-up so it only has to
	// visit every function once.
	var failed []loader.Sym
	for _, s := range ctxt.Textp {
		if sc.check(s) > limit {
			failed = append(failed, s)
		}
	}

	if len(failed) > 0 {
		// Something was over-limit, so now we do the more
		// expensive work to report a good error. First, for
		// the over-limit functions, redo the stack check but
		// record the graph this time.
		sc = newStackCheck(ctxt, true)
		for _, s := range failed {
			sc.check(s)
		}

		// Find the roots of the graph (functions that are not
		// called by any other function).
		roots := sc.findRoots()

		// Find and report all paths that go over the limit.
		// This accumulates stack depths top-down. This is
		// much less efficient because we may have to visit
		// the same function multiple times at different
		// depths, but lets us find all paths.
		for _, root := range roots {
			ctxt.Errorf(root, "nosplit stack overflow")
			chain := []stackCheckChain{{stackCheckEdge{0, root}, false}}
			sc.report(root, limit, &chain)
		}
	}
}

func newStackCheck(ctxt *Link, graph bool) *stackCheck {
	sc := &stackCheck{
		ctxt:      ctxt,
		ldr:       ctxt.loader,
		morestack: ctxt.loader.Lookup("runtime.morestack", 0),
		height:    make(map[loader.Sym]int16, len(ctxt.Textp)),
	}
	// Compute stack effect of a CALL operation. 0 on LR machines.
	// 1 register pushed on non-LR machines.
	if !ctxt.Arch.HasLR {
		sc.callSize = ctxt.Arch.RegSize
	}

	if graph {
		// We're going to record the call graph.
		sc.graph = make(map[loader.Sym][]stackCheckEdge)
	}

	return sc
}

func (sc *stackCheck) symName(sym loader.Sym) string {
	switch sym {
	case stackCheckIndirect:
		return "indirect"
	case 0:
		return "leaf"
	}
	return fmt.Sprintf("%s<%d>", sc.ldr.SymName(sym), sc.ldr.SymVersion(sym))
}

// check returns the stack height of sym. It populates sc.height and
// sc.graph for sym and every function in its call tree.
func (sc *stackCheck) check(sym loader.Sym) int {
	if h, ok := sc.height[sym]; ok {
		// We've already visited this symbol or we're in a cycle.
		return int(h)
	}
	// Store the sentinel so we can detect cycles.
	sc.height[sym] = stackCheckCycle
	// Compute and record the height and optionally edges.
	h, edges := sc.computeHeight(sym, *flagDebugNosplit || sc.graph != nil)
	if h > int(stackCheckCycle) { // Prevent integer overflow
		h = int(stackCheckCycle)
	}
	sc.height[sym] = int16(h)
	if sc.graph != nil {
		sc.graph[sym] = edges
	}

	if *flagDebugNosplit {
		for _, edge := range edges {
			fmt.Printf("nosplit: %s +%d", sc.symName(sym), edge.growth)
			if edge.target == 0 {
				// Local stack growth or leaf function.
				fmt.Printf("\n")
			} else {
				fmt.Printf(" -> %s\n", sc.symName(edge.target))
			}
		}
	}

	return h
}

// computeHeight returns the stack height of sym. If graph is true, it
// also returns the out-edges of sym.
//
// Caching is applied to this in check. Call check instead of calling
// this directly.
func (sc *stackCheck) computeHeight(sym loader.Sym, graph bool) (int, []stackCheckEdge) {
	ldr := sc.ldr

	// Check special cases.
	if sym == sc.morestack {
		// morestack looks like it calls functions, but they
		// either happen only when already on the system stack
		// (where there is ~infinite space), or after
		// switching to the system stack. Hence, its stack
		// height on the user stack is 0.
		return 0, nil
	}
	if sym == stackCheckIndirect {
		// Assume that indirect/closure calls are always to
		// splittable functions, so they just need enough room
		// to call morestack.
		return sc.callSize, []stackCheckEdge{{sc.callSize, sc.morestack}}
	}

	// Ignore calls to external functions. Assume that these calls
	// are only ever happening on the system stack, where there's
	// plenty of room.
	if ldr.AttrExternal(sym) {
		return 0, nil
	}
	if info := ldr.FuncInfo(sym); !info.Valid() { // also external
		return 0, nil
	}

	// Track the maximum height of this function and, if we're
	// recording the graph, its out-edges.
	var edges []stackCheckEdge
	maxHeight := 0
	ctxt := sc.ctxt
	// addEdge adds a stack growth out of this function to
	// function "target" or, if target == 0, a local stack growth
	// within the function.
	addEdge := func(growth int, target loader.Sym) {
		if graph {
			edges = append(edges, stackCheckEdge{growth, target})
		}
		height := growth
		if target != 0 { // Don't walk into the leaf "edge"
			height += sc.check(target)
		}
		if height > maxHeight {
			maxHeight = height
		}
	}

	if !ldr.IsNoSplit(sym) {
		// Splittable functions start with a call to
		// morestack, after which their height is 0. Account
		// for the height of the call to morestack.
		addEdge(sc.callSize, sc.morestack)
		return maxHeight, edges
	}

	// This function is nosplit, so it adjusts SP without a split
	// check.
	//
	// Walk through SP adjustments in function, consuming relocs
	// and following calls.
	maxLocalHeight := 0
	relocs, ri := ldr.Relocs(sym), 0
	pcsp := obj.NewPCIter(uint32(ctxt.Arch.MinLC))
	for pcsp.Init(ldr.Data(ldr.Pcsp(sym))); !pcsp.Done; pcsp.Next() {
		// pcsp.value is in effect for [pcsp.pc, pcsp.nextpc).
		height := int(pcsp.Value)
		if height > maxLocalHeight {
			maxLocalHeight = height
		}

		// Process calls in this span.
		for ; ri < relocs.Count(); ri++ {
			r := relocs.At(ri)
			if uint32(r.Off()) >= pcsp.NextPC {
				break
			}
			t := r.Type()
			if t.IsDirectCall() || t == objabi.R_CALLIND {
				growth := height + sc.callSize
				var target loader.Sym
				if t == objabi.R_CALLIND {
					target = stackCheckIndirect
				} else {
					target = r.Sym()
				}
				addEdge(growth, target)
			}
		}
	}
	if maxLocalHeight > maxHeight {
		// This is either a leaf function, or the function
		// grew its stack to larger than the maximum call
		// height between calls. Either way, record that local
		// stack growth.
		addEdge(maxLocalHeight, 0)
	}

	return maxHeight, edges
}

func (sc *stackCheck) findRoots() []loader.Sym {
	// Collect all nodes.
	nodes := make(map[loader.Sym]struct{})
	for k := range sc.graph {
		nodes[k] = struct{}{}
	}

	// Start a DFS from each node and delete all reachable
	// children. If we encounter an unrooted cycle, this will
	// delete everything in that cycle, so we detect this case and
	// track the lowest-numbered node encountered in the cycle and
	// put that node back as a root.
	var walk func(origin, sym loader.Sym) (cycle bool, lowest loader.Sym)
	walk = func(origin, sym loader.Sym) (cycle bool, lowest loader.Sym) {
		if _, ok := nodes[sym]; !ok {
			// We already deleted this node.
			return false, 0
		}
		delete(nodes, sym)

		if origin == sym {
			// We found an unrooted cycle. We already
			// deleted all children of this node. Walk
			// back up, tracking the lowest numbered
			// symbol in this cycle.
			return true, sym
		}

		// Delete children of this node.
		for _, out := range sc.graph[sym] {
			if c, l := walk(origin, out.target); c {
				cycle = true
				if lowest == 0 {
					// On first cycle detection,
					// add sym to the set of
					// lowest-numbered candidates.
					lowest = sym
				}
				if l < lowest {
					lowest = l
				}
			}
		}
		return
	}
	for k := range nodes {
		// Delete all children of k.
		for _, out := range sc.graph[k] {
			if cycle, lowest := walk(k, out.target); cycle {
				// This is an unrooted cycle so we
				// just deleted everything. Put back
				// the lowest-numbered symbol.
				nodes[lowest] = struct{}{}
			}
		}
	}

	// Sort roots by height. This makes the result deterministic
	// and also improves the error reporting.
	var roots []loader.Sym
	for k := range nodes {
		roots = append(roots, k)
	}
	sort.Slice(roots, func(i, j int) bool {
		h1, h2 := sc.height[roots[i]], sc.height[roots[j]]
		if h1 != h2 {
			return h1 > h2
		}
		// Secondary sort by Sym.
		return roots[i] < roots[j]
	})
	return roots
}

type stackCheckChain struct {
	stackCheckEdge
	printed bool
}

func (sc *stackCheck) report(sym loader.Sym, depth int, chain *[]stackCheckChain) {
	// Walk the out-edges of sym. We temporarily pull the edges
	// out of the graph to detect cycles and prevent infinite
	// recursion.
	edges, ok := sc.graph[sym]
	isCycle := !(ok || sym == 0)
	delete(sc.graph, sym)
	for _, out := range edges {
		*chain = append(*chain, stackCheckChain{out, false})
		sc.report(out.target, depth-out.growth, chain)
		*chain = (*chain)[:len(*chain)-1]
	}
	sc.graph[sym] = edges

	// If we've reached the end of a chain and it went over the
	// stack limit or was a cycle that would eventually go over,
	// print the whole chain.
	//
	// We should either be in morestack (which has no out-edges)
	// or the sentinel 0 Sym "called" from a leaf function (which
	// has no out-edges), or we came back around a cycle (possibly
	// to ourselves) and edges was temporarily nil'd.
	if len(edges) == 0 && (depth < 0 || isCycle) {
		var indent string
		for i := range *chain {
			ent := &(*chain)[i]
			if ent.printed {
				// Already printed on an earlier part
				// of this call tree.
				continue
			}
			ent.printed = true

			if i == 0 {
				// chain[0] is just the root function,
				// not a stack growth.
				fmt.Printf("%s\n", sc.symName(ent.target))
				continue
			}

			indent = strings.Repeat("    ", i)
			fmt.Print(indent)
			// Grows the stack X bytes and (maybe) calls Y.
			fmt.Printf("grows %d bytes", ent.growth)
			if ent.target == 0 {
				// Not a call, just a leaf. Print nothing.
			} else {
				fmt.Printf(", calls %s", sc.symName(ent.target))
			}
			fmt.Printf("\n")
		}
		// Print how far over this chain went.
		if isCycle {
			fmt.Printf("%sinfinite cycle\n", indent)
		} else {
			fmt.Printf("%s%d bytes over limit\n", indent, -depth)
		}
	}
}
