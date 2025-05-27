// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/internal/src"
	"fmt"
	"math/bits"
	"strings"
)

// walkAll computes the minimal dereferences between all pairs of
// locations.
func (b *batch) walkAll() {
	// We use a work queue to keep track of locations that we need
	// to visit, and repeatedly walk until we reach a fixed point.
	//
	// We walk once from each location (including the heap), and
	// then re-enqueue each location on its transition from
	// !persists->persists and !escapes->escapes, which can each
	// happen at most once. So we take Θ(len(e.allLocs)) walks.

	// Queue of locations to walk. Has enough room for b.allLocs
	// plus b.heapLoc, b.mutatorLoc, b.calleeLoc.
	todo := newQueue(len(b.allLocs) + 3)

	enqueue := func(loc *location) {
		if !loc.queuedWalkAll {
			loc.queuedWalkAll = true
			if loc.hasAttr(attrEscapes) {
				// Favor locations that escape to the heap,
				// which in some cases allows attrEscape to
				// propagate faster.
				todo.pushFront(loc)
			} else {
				todo.pushBack(loc)
			}
		}
	}

	for _, loc := range b.allLocs {
		todo.pushFront(loc)
		// TODO(thepudds): clean up setting queuedWalkAll.
		loc.queuedWalkAll = true
	}
	todo.pushFront(&b.mutatorLoc)
	todo.pushFront(&b.calleeLoc)
	todo.pushFront(&b.heapLoc)

	b.mutatorLoc.queuedWalkAll = true
	b.calleeLoc.queuedWalkAll = true
	b.heapLoc.queuedWalkAll = true

	var walkgen uint32
	for todo.len() > 0 {
		root := todo.popFront()
		root.queuedWalkAll = false
		walkgen++
		b.walkOne(root, walkgen, enqueue)
	}
}

// walkOne computes the minimal number of dereferences from root to
// all other locations.
func (b *batch) walkOne(root *location, walkgen uint32, enqueue func(*location)) {
	// The data flow graph has negative edges (from addressing
	// operations), so we use the Bellman-Ford algorithm. However,
	// we don't have to worry about infinite negative cycles since
	// we bound intermediate dereference counts to 0.

	root.walkgen = walkgen
	root.derefs = 0
	root.dst = nil

	if root.hasAttr(attrCalls) {
		if clo, ok := root.n.(*ir.ClosureExpr); ok {
			if fn := clo.Func; b.inMutualBatch(fn.Nname) && !fn.ClosureResultsLost() {
				fn.SetClosureResultsLost(true)

				// Re-flow from the closure's results, now that we're aware
				// we lost track of them.
				for _, result := range fn.Type().Results() {
					enqueue(b.oldLoc(result.Nname.(*ir.Name)))
				}
			}
		}
	}

	todo := newQueue(1)
	todo.pushFront(root)

	for todo.len() > 0 {
		l := todo.popFront()
		l.queuedWalkOne = 0 // no longer queued for walkOne

		derefs := l.derefs
		var newAttrs locAttr

		// If l.derefs < 0, then l's address flows to root.
		addressOf := derefs < 0
		if addressOf {
			// For a flow path like "root = &l; l = x",
			// l's address flows to root, but x's does
			// not. We recognize this by lower bounding
			// derefs at 0.
			derefs = 0

			// If l's address flows somewhere that
			// outlives it, then l needs to be heap
			// allocated.
			if b.outlives(root, l) {
				if !l.hasAttr(attrEscapes) && (logopt.Enabled() || base.Flag.LowerM >= 2) {
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: %v escapes to heap in %v:\n", base.FmtPos(l.n.Pos()), l.n, ir.FuncName(l.curfn))
					}
					explanation := b.explainPath(root, l)
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "escape", "escape", ir.FuncName(e_curfn), fmt.Sprintf("%v escapes to heap", l.n), explanation)
					}
				}
				newAttrs |= attrEscapes | attrPersists | attrMutates | attrCalls
			} else
			// If l's address flows to a persistent location, then l needs
			// to persist too.
			if root.hasAttr(attrPersists) {
				newAttrs |= attrPersists
			}
		}

		if derefs == 0 {
			newAttrs |= root.attrs & (attrMutates | attrCalls)
		}

		// l's value flows to root. If l is a function
		// parameter and root is the heap or a
		// corresponding result parameter, then record
		// that value flow for tagging the function
		// later.
		if l.param {
			if b.outlives(root, l) {
				if !l.hasAttr(attrEscapes) && (logopt.Enabled() || base.Flag.LowerM >= 2) {
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: parameter %v leaks to %s for %v with derefs=%d:\n", base.FmtPos(l.n.Pos()), l.n, b.explainLoc(root), ir.FuncName(l.curfn), derefs)
					}
					explanation := b.explainPath(root, l)
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "leak", "escape", ir.FuncName(e_curfn),
							fmt.Sprintf("parameter %v leaks to %s with derefs=%d", l.n, b.explainLoc(root), derefs), explanation)
					}
				}
				l.leakTo(root, derefs)
			}
			if root.hasAttr(attrMutates) {
				l.paramEsc.AddMutator(derefs)
			}
			if root.hasAttr(attrCalls) {
				l.paramEsc.AddCallee(derefs)
			}
		}

		if newAttrs&^l.attrs != 0 {
			l.attrs |= newAttrs
			enqueue(l)
			if l.attrs&attrEscapes != 0 {
				continue
			}
		}

		for i, edge := range l.edges {
			if edge.src.hasAttr(attrEscapes) {
				continue
			}
			d := derefs + edge.derefs
			if edge.src.walkgen != walkgen || edge.src.derefs > d {
				edge.src.walkgen = walkgen
				edge.src.derefs = d
				edge.src.dst = l
				edge.src.dstEdgeIdx = i
				// Check if already queued in todo.
				if edge.src.queuedWalkOne != walkgen {
					edge.src.queuedWalkOne = walkgen // Mark queued for this walkgen.

					// Place at the back to possibly give time for
					// other possible attribute changes to src.
					todo.pushBack(edge.src)
				}
			}
		}
	}
}

// explainPath prints an explanation of how src flows to the walk root.
func (b *batch) explainPath(root, src *location) []*logopt.LoggedOpt {
	visited := make(map[*location]bool)
	pos := base.FmtPos(src.n.Pos())
	var explanation []*logopt.LoggedOpt
	for {
		// Prevent infinite loop.
		if visited[src] {
			if base.Flag.LowerM >= 2 {
				fmt.Printf("%s:   warning: truncated explanation due to assignment cycle; see golang.org/issue/35518\n", pos)
			}
			break
		}
		visited[src] = true
		dst := src.dst
		edge := &dst.edges[src.dstEdgeIdx]
		if edge.src != src {
			base.Fatalf("path inconsistency: %v != %v", edge.src, src)
		}

		explanation = b.explainFlow(pos, dst, src, edge.derefs, edge.notes, explanation)

		if dst == root {
			break
		}
		src = dst
	}

	return explanation
}

func (b *batch) explainFlow(pos string, dst, srcloc *location, derefs int, notes *note, explanation []*logopt.LoggedOpt) []*logopt.LoggedOpt {
	ops := "&"
	if derefs >= 0 {
		ops = strings.Repeat("*", derefs)
	}
	print := base.Flag.LowerM >= 2

	flow := fmt.Sprintf("   flow: %s ← %s%v:", b.explainLoc(dst), ops, b.explainLoc(srcloc))
	if print {
		fmt.Printf("%s:%s\n", pos, flow)
	}
	if logopt.Enabled() {
		var epos src.XPos
		if notes != nil {
			epos = notes.where.Pos()
		} else if srcloc != nil && srcloc.n != nil {
			epos = srcloc.n.Pos()
		}
		var e_curfn *ir.Func // TODO(mdempsky): Fix.
		explanation = append(explanation, logopt.NewLoggedOpt(epos, epos, "escflow", "escape", ir.FuncName(e_curfn), flow))
	}

	for note := notes; note != nil; note = note.next {
		if print {
			fmt.Printf("%s:     from %v (%v) at %s\n", pos, note.where, note.why, base.FmtPos(note.where.Pos()))
		}
		if logopt.Enabled() {
			var e_curfn *ir.Func // TODO(mdempsky): Fix.
			notePos := note.where.Pos()
			explanation = append(explanation, logopt.NewLoggedOpt(notePos, notePos, "escflow", "escape", ir.FuncName(e_curfn),
				fmt.Sprintf("     from %v (%v)", note.where, note.why)))
		}
	}
	return explanation
}

func (b *batch) explainLoc(l *location) string {
	if l == &b.heapLoc {
		return "{heap}"
	}
	if l.n == nil {
		// TODO(mdempsky): Omit entirely.
		return "{temp}"
	}
	if l.n.Op() == ir.ONAME {
		return fmt.Sprintf("%v", l.n)
	}
	return fmt.Sprintf("{storage for %v}", l.n)
}

// outlives reports whether values stored in l may survive beyond
// other's lifetime if stack allocated.
func (b *batch) outlives(l, other *location) bool {
	// The heap outlives everything.
	if l.hasAttr(attrEscapes) {
		return true
	}

	// Pseudo-locations that don't really exist.
	if l == &b.mutatorLoc || l == &b.calleeLoc {
		return false
	}

	// We don't know what callers do with returned values, so
	// pessimistically we need to assume they flow to the heap and
	// outlive everything too.
	if l.paramOut {
		// Exception: Closures can return locations allocated outside of
		// them without forcing them to the heap, if we can statically
		// identify all call sites. For example:
		//
		//	var u int  // okay to stack allocate
		//	fn := func() *int { return &u }()
		//	*fn() = 42
		if ir.ContainsClosure(other.curfn, l.curfn) && !l.curfn.ClosureResultsLost() {
			return false
		}

		return true
	}

	// If l and other are within the same function, then l
	// outlives other if it was declared outside other's loop
	// scope. For example:
	//
	//	var l *int
	//	for {
	//		l = new(int) // must heap allocate: outlives for loop
	//	}
	if l.curfn == other.curfn && l.loopDepth < other.loopDepth {
		return true
	}

	// If other is declared within a child closure of where l is
	// declared, then l outlives it. For example:
	//
	//	var l *int
	//	func() {
	//		l = new(int) // must heap allocate: outlives call frame (if not inlined)
	//	}()
	if ir.ContainsClosure(l.curfn, other.curfn) {
		return true
	}

	return false
}

// queue implements a queue of locations for use in WalkAll and WalkOne.
// It supports pushing to front & back, and popping from front.
// TODO(thepudds): does cmd/compile have a deque or similar somewhere?
type queue struct {
	locs  []*location
	head  int // index of front element
	tail  int // next back element
	elems int
}

func newQueue(capacity int) *queue {
	capacity = max(capacity, 2)
	capacity = 1 << bits.Len64(uint64(capacity-1)) // round up to a power of 2
	return &queue{locs: make([]*location, capacity)}
}

// pushFront adds an element to the front of the queue.
func (q *queue) pushFront(loc *location) {
	if q.elems == len(q.locs) {
		q.grow()
	}
	q.head = q.wrap(q.head - 1)
	q.locs[q.head] = loc
	q.elems++
}

// pushBack adds an element to the back of the queue.
func (q *queue) pushBack(loc *location) {
	if q.elems == len(q.locs) {
		q.grow()
	}
	q.locs[q.tail] = loc
	q.tail = q.wrap(q.tail + 1)
	q.elems++
}

// popFront removes the front of the queue.
func (q *queue) popFront() *location {
	if q.elems == 0 {
		return nil
	}
	loc := q.locs[q.head]
	q.head = q.wrap(q.head + 1)
	q.elems--
	return loc
}

// grow doubles the capacity.
func (q *queue) grow() {
	newLocs := make([]*location, len(q.locs)*2)
	for i := range q.elems {
		// Copy over our elements in order.
		newLocs[i] = q.locs[q.wrap(q.head+i)]
	}
	q.locs = newLocs
	q.head = 0
	q.tail = q.elems
}

func (q *queue) len() int       { return q.elems }
func (q *queue) wrap(i int) int { return i & (len(q.locs) - 1) }
