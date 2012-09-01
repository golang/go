// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exp/locale/collate"
	"fmt"
	"log"
	"sort"
	"strings"
	"unicode"
)

type logicalAnchor int

const (
	firstAnchor logicalAnchor = -1
	noAnchor                  = 0
	lastAnchor                = 1
)

// TODO: move type entry from builder.go to this file.

// nextIndexed gets the next entry that needs to be stored in the table.
// It returns the entry and the collation level at which the next entry differs
// from the current entry.
// Entries that can be explicitly derived and logical reset positions are
// examples of entries that will not be indexed.
func (e *entry) nextIndexed() (*entry, collate.Level) {
	level := e.level
	for e = e.next; e != nil && e.exclude; e = e.next {
		if e.level < level {
			level = e.level
		}
	}
	return e, level
}

// remove unlinks entry e from the sorted chain and clears the collation
// elements. e may not be at the front or end of the list. This should always
// be the case, as the front and end of the list are always logical anchors,
// which may not be removed.
func (e *entry) remove() {
	if e.logical != noAnchor {
		log.Fatalf("may not remove anchor %q", e.str)
	}
	if e.prev != nil {
		e.prev.next = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	}
	e.elems = nil
}

// insertAfter inserts t after e.
func (e *entry) insertAfter(n *entry) {
	if e == n {
		panic("e == anchor")
	}
	if e == nil {
		panic("unexpected nil anchor")
	}
	n.remove()
	n.decompose = false // redo decomposition test

	n.next = e.next
	n.prev = e
	e.next.prev = n
	e.next = n
}

// entryLess returns true if a sorts before b and false otherwise.
func entryLess(a, b *entry) bool {
	if res, _ := compareWeights(a.elems, b.elems); res != 0 {
		return res == -1
	}
	if a.logical != noAnchor {
		return a.logical == firstAnchor
	}
	if b.logical != noAnchor {
		return b.logical == lastAnchor
	}
	return a.str < b.str
}

type sortedEntries []*entry

func (s sortedEntries) Len() int {
	return len(s)
}

func (s sortedEntries) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s sortedEntries) Less(i, j int) bool {
	return entryLess(s[i], s[j])
}

type ordering struct {
	entryMap map[string]*entry
	ordered  []*entry
}

// insert inserts e into both entryMap and ordered.
// Note that insert simply appends e to ordered.  To reattain a sorted
// order, o.sort() should be called.
func (o *ordering) insert(e *entry) {
	o.entryMap[e.str] = e
	o.ordered = append(o.ordered, e)
}

// newEntry creates a new entry for the given info and inserts it into
// the index.
func (o *ordering) newEntry(s string, ces [][]int) *entry {
	e := &entry{
		runes: []rune(s),
		elems: ces,
		str:   s,
	}
	o.insert(e)
	return e
}

// find looks up and returns the entry for the given string.
// It returns nil if str is not in the index and if an implicit value
// cannot be derived, that is, if str represents more than one rune.
func (o *ordering) find(str string) *entry {
	e := o.entryMap[str]
	if e == nil {
		r := []rune(str)
		if len(r) == 1 {
			e = o.newEntry(string(r[0]), [][]int{
				[]int{
					implicitPrimary(r[0]),
					defaultSecondary,
					defaultTertiary,
					int(r[0]),
				},
			})
			e.exclude = true // do not index implicits
		}
	}
	return e
}

// makeRootOrdering returns a newly initialized ordering value and populates
// it with a set of logical reset points that can be used as anchors.
// The anchors first_tertiary_ignorable and __END__ will always sort at
// the beginning and end, respectively. This means that prev and next are non-nil
// for any indexed entry.
func makeRootOrdering() ordering {
	const max = unicode.MaxRune
	o := ordering{
		entryMap: make(map[string]*entry),
	}
	insert := func(typ logicalAnchor, s string, ce []int) {
		// Use key format as used in UCA rules.
		e := o.newEntry(fmt.Sprintf("[%s]", s), [][]int{ce})
		// Also add index entry for XML format.
		o.entryMap[fmt.Sprintf("<%s/>", strings.Replace(s, " ", "_", -1))] = e
		e.runes = nil
		e.exclude = true
		e.logical = typ
	}
	insert(firstAnchor, "first tertiary ignorable", []int{0, 0, 0, 0})
	insert(lastAnchor, "last tertiary ignorable", []int{0, 0, 0, max})
	insert(lastAnchor, "last primary ignorable", []int{0, defaultSecondary, defaultTertiary, max})
	insert(lastAnchor, "last non ignorable", []int{maxPrimary, defaultSecondary, defaultTertiary, max})
	insert(lastAnchor, "__END__", []int{1 << maxPrimaryBits, defaultSecondary, defaultTertiary, max})
	return o
}

// clone copies all ordering of es into a new ordering value.
func (o *ordering) clone() *ordering {
	o.sort()
	oo := ordering{
		entryMap: make(map[string]*entry),
	}
	for _, e := range o.ordered {
		ne := &entry{
			runes:     e.runes,
			elems:     e.elems,
			str:       e.str,
			decompose: e.decompose,
			exclude:   e.exclude,
		}
		oo.insert(ne)
	}
	oo.sort() // link all ordering.
	return &oo
}

// front returns the first entry to be indexed.
// It assumes that sort() has been called.
func (o *ordering) front() *entry {
	e := o.ordered[0]
	if e.prev != nil {
		log.Panicf("unexpected first entry: %v", e)
	}
	// The first entry is always a logical position, which should not be indexed.
	e, _ = e.nextIndexed()
	return e
}

// sort sorts all ordering based on their collation elements and initializes
// the prev, next, and level fields accordingly.
func (o *ordering) sort() {
	sort.Sort(sortedEntries(o.ordered))
	l := o.ordered
	for i := 1; i < len(l); i++ {
		k := i - 1
		l[k].next = l[i]
		_, l[k].level = compareWeights(l[k].elems, l[i].elems)
		l[i].prev = l[k]
	}
}

// genColElems generates a collation element array from the runes in str. This
// assumes that all collation elements have already been added to the Builder.
func (o *ordering) genColElems(str string) [][]int {
	elems := [][]int{}
	for _, r := range []rune(str) {
		elems = append(elems, o.find(string(r)).elems...)
	}
	return elems
}
