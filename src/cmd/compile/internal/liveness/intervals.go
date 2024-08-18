// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package liveness

// This file defines an "Intervals" helper type that stores a
// sorted sequence of disjoint ranges or intervals. An Intervals
// example: { [0,5) [9-12) [100,101) }, which corresponds to the
// numbers 0-4, 9-11, and 100. Once an Intervals object is created, it
// can be tested to see if it has any overlap with another Intervals
// object, or it can be merged with another Intervals object to form a
// union of the two.
//
// The intended use case for this helper is in describing object or
// variable lifetime ranges within a linearized program representation
// where each IR instruction has a slot or index. Example:
//
//          b1:
//  0        VarDef abc
//  1        memset(abc,0)
//  2        VarDef xyz
//  3        memset(xyz,0)
//  4        abc.f1 = 2
//  5        xyz.f3 = 9
//  6        if q goto B4
//  7 B3:    z = xyz.x
//  8        goto B5
//  9 B4:    z = abc.x
//           // fallthrough
// 10 B5:    z++
//
// To describe the lifetime of the variables above we might use these
// intervals:
//
//    "abc"   [1,7), [9,10)
//    "xyz"   [3,8)
//
// Clients can construct an Intervals object from a given IR sequence
// using the "IntervalsBuilder" helper abstraction (one builder per
// candidate variable), by making a
// backwards sweep and invoking the Live/Kill methods to note the
// starts and end of a given lifetime. For the example above, we would
// expect to see this sequence of calls to Live/Kill:
//
//    abc:  Live(9), Kill(8), Live(6), Kill(0)
//    xyz:  Live(8), Kill(2)

import (
	"fmt"
	"os"
	"strings"
)

const debugtrace = false

// Interval hols the range [st,en).
type Interval struct {
	st, en int
}

// Intervals is a sequence of sorted, disjoint intervals.
type Intervals []Interval

func (i Interval) String() string {
	return fmt.Sprintf("[%d,%d)", i.st, i.en)
}

// TEMPORARY until bootstrap version catches up.
func imin(i, j int) int {
	if i < j {
		return i
	}
	return j
}

// TEMPORARY until bootstrap version catches up.
func imax(i, j int) int {
	if i > j {
		return i
	}
	return j
}

// Overlaps returns true if here is any overlap between i and i2.
func (i Interval) Overlaps(i2 Interval) bool {
	return (imin(i.en, i2.en) - imax(i.st, i2.st)) > 0
}

// adjacent returns true if the start of one interval is equal to the
// end of another interval (e.g. they represent consecutive ranges).
func (i1 Interval) adjacent(i2 Interval) bool {
	return i1.en == i2.st || i2.en == i1.st
}

// MergeInto merges interval i2 into i1. This version happens to
// require that the two intervals either overlap or are adjacent.
func (i1 *Interval) MergeInto(i2 Interval) error {
	if !i1.Overlaps(i2) && !i1.adjacent(i2) {
		return fmt.Errorf("merge method invoked on non-overlapping/non-adjacent")
	}
	i1.st = imin(i1.st, i2.st)
	i1.en = imax(i1.en, i2.en)
	return nil
}

// IntervalsBuilder is a helper for constructing intervals based on
// live dataflow sets for a series of BBs where we're making a
// backwards pass over each BB looking for uses and kills. The
// expected use case is:
//
//   - invoke MakeIntervalsBuilder to create a new object "b"
//   - series of calls to b.Live/b.Kill based on a backwards reverse layout
//     order scan over instructions
//   - invoke b.Finish() to produce final set
//
// See the Live method comment for an IR example.
type IntervalsBuilder struct {
	s Intervals
	// index of last instruction visited plus 1
	lidx int
}

func (c *IntervalsBuilder) last() int {
	return c.lidx - 1
}

func (c *IntervalsBuilder) setLast(x int) {
	c.lidx = x + 1
}

func (c *IntervalsBuilder) Finish() (Intervals, error) {
	// Reverse intervals list and check.
	// FIXME: replace with slices.Reverse once the
	// bootstrap version supports it.
	for i, j := 0, len(c.s)-1; i < j; i, j = i+1, j-1 {
		c.s[i], c.s[j] = c.s[j], c.s[i]
	}
	if err := check(c.s); err != nil {
		return Intervals{}, err
	}
	r := c.s
	return r, nil
}

// Live method should be invoked on instruction at position p if instr
// contains an upwards-exposed use of a resource. See the example in
// the comment at the beginning of this file for an example.
func (c *IntervalsBuilder) Live(pos int) error {
	if pos < 0 {
		return fmt.Errorf("bad pos, negative")
	}
	if c.last() == -1 {
		c.setLast(pos)
		if debugtrace {
			fmt.Fprintf(os.Stderr, "=-= begin lifetime at pos=%d\n", pos)
		}
		c.s = append(c.s, Interval{st: pos, en: pos + 1})
		return nil
	}
	if pos >= c.last() {
		return fmt.Errorf("pos not decreasing")
	}
	// extend lifetime across this pos
	c.s[len(c.s)-1].st = pos
	c.setLast(pos)
	return nil
}

// Kill method should be invoked on instruction at position p if instr
// should be treated as having a kill (lifetime end) for the
// resource. See the example in the comment at the beginning of this
// file for an example. Note that if we see a kill at position K for a
// resource currently live since J, this will result in a lifetime
// segment of [K+1,J+1), the assumption being that the first live
// instruction will be the one after the kill position, not the kill
// position itself.
func (c *IntervalsBuilder) Kill(pos int) error {
	if pos < 0 {
		return fmt.Errorf("bad pos, negative")
	}
	if c.last() == -1 {
		return nil
	}
	if pos >= c.last() {
		return fmt.Errorf("pos not decreasing")
	}
	c.s[len(c.s)-1].st = pos + 1
	// terminate lifetime
	c.setLast(-1)
	if debugtrace {
		fmt.Fprintf(os.Stderr, "=-= term lifetime at pos=%d\n", pos)
	}
	return nil
}

// check examines the intervals in "is" to try to find internal
// inconsistencies or problems.
func check(is Intervals) error {
	for i := 0; i < len(is); i++ {
		st := is[i].st
		en := is[i].en
		if en <= st {
			return fmt.Errorf("bad range elem %d:%d, en<=st", st, en)
		}
		if i == 0 {
			continue
		}
		// check for badly ordered starts
		pst := is[i-1].st
		pen := is[i-1].en
		if pst >= st {
			return fmt.Errorf("range start not ordered %d:%d less than prev %d:%d", st, en,
				pst, pen)
		}
		// check end of last range against start of this range
		if pen > st {
			return fmt.Errorf("bad range elem %d:%d overlaps prev %d:%d", st, en,
				pst, pen)
		}
	}
	return nil
}

func (is *Intervals) String() string {
	var sb strings.Builder
	for i := range *is {
		if i != 0 {
			sb.WriteString(" ")
		}
		sb.WriteString((*is)[i].String())
	}
	return sb.String()
}

// intWithIdx holds an interval i and an index pairIndex storing i's
// position (either 0 or 1) within some previously specified interval
// pair <I1,I2>; a pairIndex of -1 is used to signal "end of
// iteration". Used for Intervals operations, not expected to be
// exported.
type intWithIdx struct {
	i         Interval
	pairIndex int
}

func (iwi intWithIdx) done() bool {
	return iwi.pairIndex == -1
}

// pairVisitor provides a way to visit (iterate through) each interval
// within a pair of Intervals in order of increasing start time. Expected
// usage model:
//
//	func example(i1, i2 Intervals) {
//	  var pairVisitor pv
//	  cur := pv.init(i1, i2);
//	  for !cur.done() {
//	     fmt.Printf("interval %s from i%d", cur.i.String(), cur.pairIndex+1)
//	     cur = pv.nxt()
//	  }
//	}
//
// Used internally for Intervals operations, not expected to be exported.
type pairVisitor struct {
	cur    intWithIdx
	i1pos  int
	i2pos  int
	i1, i2 Intervals
}

// init initializes a pairVisitor for the specified pair of intervals
// i1 and i2 and returns an intWithIdx object that points to the first
// interval by start position within i1/i2.
func (pv *pairVisitor) init(i1, i2 Intervals) intWithIdx {
	pv.i1, pv.i2 = i1, i2
	pv.cur = pv.sel()
	return pv.cur
}

// nxt advances the pairVisitor to the next interval by starting
// position within the pair, returning an intWithIdx that describes
// the interval.
func (pv *pairVisitor) nxt() intWithIdx {
	if pv.cur.pairIndex == 0 {
		pv.i1pos++
	} else {
		pv.i2pos++
	}
	pv.cur = pv.sel()
	return pv.cur
}

// sel is a helper function used by 'init' and 'nxt' above; it selects
// the earlier of the two intervals at the current positions within i1
// and i2, or a degenerate (pairIndex -1) intWithIdx if we have no
// more intervals to visit.
func (pv *pairVisitor) sel() intWithIdx {
	var c1, c2 intWithIdx
	if pv.i1pos >= len(pv.i1) {
		c1.pairIndex = -1
	} else {
		c1 = intWithIdx{i: pv.i1[pv.i1pos], pairIndex: 0}
	}
	if pv.i2pos >= len(pv.i2) {
		c2.pairIndex = -1
	} else {
		c2 = intWithIdx{i: pv.i2[pv.i2pos], pairIndex: 1}
	}
	if c1.pairIndex == -1 {
		return c2
	}
	if c2.pairIndex == -1 {
		return c1
	}
	if c1.i.st <= c2.i.st {
		return c1
	}
	return c2
}

// Overlaps returns whether any of the component ranges in is overlaps
// with some range in is2.
func (is Intervals) Overlaps(is2 Intervals) bool {
	// check for empty intervals
	if len(is) == 0 || len(is2) == 0 {
		return false
	}
	li := len(is)
	li2 := len(is2)
	// check for completely disjoint ranges
	if is[li-1].en <= is2[0].st ||
		is[0].st >= is2[li2-1].en {
		return false
	}
	// walk the combined sets of intervals and check for piecewise
	// overlap.
	var pv pairVisitor
	first := pv.init(is, is2)
	for {
		second := pv.nxt()
		if second.done() {
			break
		}
		if first.pairIndex == second.pairIndex {
			first = second
			continue
		}
		if first.i.Overlaps(second.i) {
			return true
		}
		first = second
	}
	return false
}

// Merge combines the intervals from "is" and "is2" and returns
// a new Intervals object containing all combined ranges from the
// two inputs.
func (is Intervals) Merge(is2 Intervals) Intervals {
	if len(is) == 0 {
		return is2
	} else if len(is2) == 0 {
		return is
	}
	// walk the combined set of intervals and merge them together.
	var ret Intervals
	var pv pairVisitor
	cur := pv.init(is, is2)
	for {
		second := pv.nxt()
		if second.done() {
			break
		}

		// Check for overlap between cur and second. If no overlap
		// then add cur to result and move on.
		if !cur.i.Overlaps(second.i) && !cur.i.adjacent(second.i) {
			ret = append(ret, cur.i)
			cur = second
			continue
		}
		// cur overlaps with second; merge second into cur
		cur.i.MergeInto(second.i)
	}
	ret = append(ret, cur.i)
	return ret
}
