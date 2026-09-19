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

// walkState contains the root properties used by a walk. Roots with equal
// states can be analyzed together.
type walkState struct {
	sink      *location // canonical leak sink; not necessarily a walk root
	curfn     *ir.Func
	loopDepth int
	attrs     locAttr
}

func (s walkState) hasAttr(attr locAttr) bool { return s.attrs&attr != 0 }

// walkPath is an immutable path from a location to one of the roots of the
// current walk.
type walkPath struct {
	root    *location // root reached by this path
	dst     *location // destination of the first edge
	edgeIdx int       // index of the first edge in dst.edges
	next    *walkPath // path from dst to root
}

// walkState returns the normalized walk state for loc.
func (b *batch) walkState(loc *location) walkState {
	s := walkState{
		sink:      &b.heapLoc,
		curfn:     loc.curfn,
		loopDepth: loc.loopDepth,
		attrs:     loc.attrs,
	}
	if loc.paramOut || loc == &b.mutatorLoc || loc == &b.calleeLoc {
		s.sink = loc
		return s
	}
	if loc.hasAttr(attrEscapes) {
		// outlives returns true for all escaping roots.
		s.curfn = nil
		s.loopDepth = 0
	}
	return s
}

// walkAll computes the minimal dereferences from each group of roots to
// all other locations.
func (b *batch) walkAll() {
	// We use a work queue to keep track of locations that we need
	// to visit, and repeatedly walk until we reach a fixed point.
	//
	// We walk once from each group of locations with the same state, since
	// their effects depend only on the minimum dereference count from the
	// group. We re-enqueue locations when their attributes change, grouping
	// them using their new state.

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
	walkTodo := newQueue(len(b.allLocs) + 3)
	groups := make(map[walkState][]*location)
	var states []walkState
	for todo.len() > 0 {
		// Process the queue in rounds. At the start of a round, group roots
		// whose walk states are equal. Each walk uses the captured state rather
		// than the roots' live attributes, so the roots only need to agree when
		// they are grouped.
		//
		// An earlier walk may add attributes to a root scheduled for a later
		// group. The root is then re-enqueued to be walked with its new state
		// in the next round. Walking it here with its old state is still safe,
		// because attributes and the effects propagated from them only grow.
		clear(groups)
		states = states[:0]
		for todo.len() > 0 {
			root := todo.popFront()
			root.queuedWalkAll = false
			state := b.walkState(root)
			if _, ok := groups[state]; !ok {
				states = append(states, state)
			}
			groups[state] = append(groups[state], root)
		}
		for _, state := range states {
			walkgen++
			b.walk(state, groups[state], walkgen, enqueue, walkTodo)
		}
	}
}

// walk computes the minimal number of dereferences from roots that were
// scheduled with state s to all other locations. A root's live attributes may
// have grown since it was scheduled; the resulting new state is walked in a
// later round.
func (b *batch) walk(s walkState, roots []*location, walkgen uint32, enqueue func(*location), todo *queue) {
	// The data flow graph has negative edges (from addressing
	// operations), so we use the Bellman-Ford algorithm. However,
	// we don't have to worry about infinite negative cycles since
	// we bound intermediate dereference counts to 0.

	diagnose := base.Flag.LowerM >= 2 || logopt.Enabled()
	var paths map[*location]*walkPath
	if diagnose {
		paths = make(map[*location]*walkPath)
	}
	todo.reset()
	for _, r := range roots {
		r.walkgen = walkgen
		r.derefs = 0
		r.queuedWalk = walkgen
		todo.pushBack(r)

		if s.hasAttr(attrCalls) {
			if clo, ok := r.n.(*ir.ClosureExpr); ok {
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
	}

	for todo.len() > 0 {
		l := todo.popFront()
		l.queuedWalk = 0 // no longer queued for walk

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
			if s.outlives(b, l) {
				if !l.hasAttr(attrEscapes) && diagnose {
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: %v escapes to heap in %v:\n", base.FmtPos(l.n.Pos()), l.n, ir.FuncName(l.curfn))
					}
					root := walkRoot(l, paths)
					explanation := b.explainPath(root, l, paths[l])
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "escape", "escape", ir.FuncName(e_curfn), fmt.Sprintf("%v escapes to heap", l.n), explanation)
					}
				}
				newAttrs |= attrEscapes | attrPersists | attrMutates | attrCalls
			} else
			// If l's address flows to a persistent location, then l needs
			// to persist too.
			if s.hasAttr(attrPersists) {
				newAttrs |= attrPersists
			}
		}

		if derefs == 0 {
			newAttrs |= s.attrs & (attrMutates | attrCalls)
		}

		// l's value flows to root. If l is a function
		// parameter and root is the heap or a
		// corresponding result parameter, then record
		// that value flow for tagging the function
		// later.
		if l.param {
			if s.outlives(b, l) {
				if !l.hasAttr(attrEscapes) && diagnose {
					root := walkRoot(l, paths)
					if base.Flag.LowerM >= 2 {
						fmt.Printf("%s: parameter %v leaks to %s for %v with derefs=%d:\n", base.FmtPos(l.n.Pos()), l.n, b.explainLoc(root), ir.FuncName(l.curfn), derefs)
					}
					explanation := b.explainPath(root, l, paths[l])
					if logopt.Enabled() {
						var e_curfn *ir.Func // TODO(mdempsky): Fix.
						logopt.LogOpt(l.n.Pos(), "leak", "escape", ir.FuncName(e_curfn),
							fmt.Sprintf("parameter %v leaks to %s with derefs=%d", l.n, b.explainLoc(root), derefs), explanation)
					}
				}
				l.leakTo(s.sink, derefs)
			}
			if s.hasAttr(attrMutates) {
				l.paramEsc.AddMutator(derefs)
			}
			if s.hasAttr(attrCalls) {
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
				if diagnose {
					paths[edge.src] = &walkPath{
						root:    walkRoot(l, paths),
						dst:     l,
						edgeIdx: i,
						next:    paths[l],
					}
				}
				// Check if already queued in todo.
				if edge.src.queuedWalk != walkgen {
					edge.src.queuedWalk = walkgen // Mark queued for this walkgen.

					// Place at the back to possibly give time for
					// other possible attribute changes to src.
					todo.pushBack(edge.src)
				}
			}
		}
	}
}

func walkRoot(l *location, paths map[*location]*walkPath) *location {
	if path := paths[l]; path != nil {
		return path.root
	}
	return l
}

// explainPath prints an explanation of how src flows to the walk root.
func (b *batch) explainPath(root, src *location, path *walkPath) []*logopt.LoggedOpt {
	visited := make(map[*location]bool)
	pos := base.FmtPos(src.n.Pos())
	var explanation []*logopt.LoggedOpt
	for ; path != nil; path = path.next {
		// Prevent infinite loop.
		if visited[src] {
			if base.Flag.LowerM >= 2 {
				fmt.Printf("%s:   warning: truncated explanation due to assignment cycle; see golang.org/issue/35518\n", pos)
			}
			return explanation
		}
		visited[src] = true
		dst := path.dst
		edge := &dst.edges[path.edgeIdx]
		if edge.src != src {
			base.Fatalf("path inconsistency: %v != %v", edge.src, src)
		}

		explanation = b.explainFlow(pos, dst, src, edge.derefs, edge.notes, explanation)

		src = dst
	}
	if src != root {
		base.Fatalf("path root inconsistency: %v != %v", src, root)
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

// outlives reports whether values stored in roots with state s may survive
// beyond other's lifetime if stack allocated.
func (s walkState) outlives(b *batch, other *location) bool {
	// The heap outlives everything.
	if s.hasAttr(attrEscapes) {
		return true
	}

	// Pseudo-locations that don't really exist.
	if s.sink == &b.mutatorLoc || s.sink == &b.calleeLoc {
		return false
	}

	// We don't know what callers do with returned values, so
	// pessimistically we need to assume they flow to the heap and
	// outlive everything too.
	if s.sink != nil && s.sink.paramOut {
		// Exception: Closures can return locations allocated outside of
		// them without forcing them to the heap, if we can statically
		// identify all call sites. For example:
		//
		//	var u int  // okay to stack allocate
		//	fn := func() *int { return &u }()
		//	*fn() = 42
		if ir.ContainsClosure(other.curfn, s.curfn) && !s.curfn.ClosureResultsLost() {
			return false
		}

		return true
	}

	// If root and other are within the same function, then root
	// outlives other if it was declared outside other's loop
	// scope. For example:
	//
	//	var l *int
	//	for {
	//		l = new(int) // must heap allocate: outlives for loop
	//	}
	if s.curfn == other.curfn && s.loopDepth < other.loopDepth {
		return true
	}

	// If other is declared within a child closure of where root is
	// declared, then root outlives it. For example:
	//
	//	var l *int
	//	func() {
	//		l = new(int) // must heap allocate: outlives call frame (if not inlined)
	//	}()
	if ir.ContainsClosure(s.curfn, other.curfn) {
		return true
	}

	return false
}

// queue implements a queue of locations for use in walkAll and walk.
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

func (q *queue) reset() {
	q.head = 0
	q.tail = 0
	q.elems = 0
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
