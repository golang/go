// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exp/locale/collate"
	"exp/norm"
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

// entry is used to keep track of a single entry in the collation element table
// during building. Examples of entries can be found in the Default Unicode
// Collation Element Table.
// See http://www.unicode.org/Public/UCA/6.0.0/allkeys.txt.
type entry struct {
	str    string // same as string(runes)
	runes  []rune
	elems  []rawCE // the collation elements
	extend string  // weights of extend to be appended to elems
	before bool    // weights relative to next instead of previous.
	lock   bool    // entry is used in extension and can no longer be moved.

	// prev, next, and level are used to keep track of tailorings.
	prev, next *entry
	level      collate.Level // next differs at this level
	skipRemove bool          // do not unlink when removed

	decompose bool // can use NFKD decomposition to generate elems
	exclude   bool // do not include in table
	implicit  bool // derived, is not included in the list
	modified  bool // entry was modified in tailoring
	logical   logicalAnchor

	expansionIndex    int // used to store index into expansion table
	contractionHandle ctHandle
	contractionIndex  int // index into contraction elements
}

func (e *entry) String() string {
	return fmt.Sprintf("%X (%q) -> %X (ch:%x; ci:%d, ei:%d)",
		e.runes, e.str, e.elems, e.contractionHandle, e.contractionIndex, e.expansionIndex)
}

func (e *entry) skip() bool {
	return e.contraction()
}

func (e *entry) expansion() bool {
	return !e.decompose && len(e.elems) > 1
}

func (e *entry) contraction() bool {
	return len(e.runes) > 1
}

func (e *entry) contractionStarter() bool {
	return e.contractionHandle.n != 0
}

// nextIndexed gets the next entry that needs to be stored in the table.
// It returns the entry and the collation level at which the next entry differs
// from the current entry.
// Entries that can be explicitly derived and logical reset positions are
// examples of entries that will not be indexed.
func (e *entry) nextIndexed() (*entry, collate.Level) {
	level := e.level
	for e = e.next; e != nil && (e.exclude || len(e.elems) == 0); e = e.next {
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
	// TODO: need to set e.prev.level to e.level if e.level is smaller?
	e.elems = nil
	if !e.skipRemove {
		if e.prev != nil {
			e.prev.next = e.next
		}
		if e.next != nil {
			e.next.prev = e.prev
		}
	}
	e.skipRemove = false
}

// insertAfter inserts n after e.
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
	if e.next != nil {
		e.next.prev = n
	}
	e.next = n
}

// insertBefore inserts n before e.
func (e *entry) insertBefore(n *entry) {
	if e == n {
		panic("e == anchor")
	}
	if e == nil {
		panic("unexpected nil anchor")
	}
	n.remove()
	n.decompose = false // redo decomposition test

	n.prev = e.prev
	n.next = e
	if e.prev != nil {
		e.prev.next = n
	}
	e.prev = n
}

func (e *entry) encodeBase() (ce uint32, err error) {
	switch {
	case e.expansion():
		ce, err = makeExpandIndex(e.expansionIndex)
	default:
		if e.decompose {
			log.Fatal("decompose should be handled elsewhere")
		}
		ce, err = makeCE(e.elems[0])
	}
	return
}

func (e *entry) encode() (ce uint32, err error) {
	if e.skip() {
		log.Fatal("cannot build colElem for entry that should be skipped")
	}
	switch {
	case e.decompose:
		t1 := e.elems[0].w[2]
		t2 := 0
		if len(e.elems) > 1 {
			t2 = e.elems[1].w[2]
		}
		ce, err = makeDecompose(t1, t2)
	case e.contractionStarter():
		ce, err = makeContractIndex(e.contractionHandle, e.contractionIndex)
	default:
		if len(e.runes) > 1 {
			log.Fatal("colElem: contractions are handled in contraction trie")
		}
		ce, err = e.encodeBase()
	}
	return
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
	id       string
	entryMap map[string]*entry
	ordered  []*entry
	handle   *trieHandle
}

// insert inserts e into both entryMap and ordered.
// Note that insert simply appends e to ordered.  To reattain a sorted
// order, o.sort() should be called.
func (o *ordering) insert(e *entry) {
	if e.logical == noAnchor {
		o.entryMap[e.str] = e
	} else {
		// Use key format as used in UCA rules.
		o.entryMap[fmt.Sprintf("[%s]", e.str)] = e
		// Also add index entry for XML format.
		o.entryMap[fmt.Sprintf("<%s/>", strings.Replace(e.str, " ", "_", -1))] = e
	}
	o.ordered = append(o.ordered, e)
}

// newEntry creates a new entry for the given info and inserts it into
// the index.
func (o *ordering) newEntry(s string, ces []rawCE) *entry {
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
			const (
				firstHangul = 0xAC00
				lastHangul  = 0xD7A3
			)
			if r[0] >= firstHangul && r[0] <= lastHangul {
				ce := []rawCE{}
				nfd := norm.NFD.String(str)
				for _, r := range nfd {
					ce = append(ce, o.find(string(r)).elems...)
				}
				e = o.newEntry(nfd, ce)
			} else {
				e = o.newEntry(string(r[0]), []rawCE{
					{w: []int{
						implicitPrimary(r[0]),
						defaultSecondary,
						defaultTertiary,
						int(r[0]),
					},
					},
				})
				e.modified = true
			}
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
		e := &entry{
			elems:   []rawCE{{w: ce}},
			str:     s,
			exclude: true,
			logical: typ,
		}
		o.insert(e)
	}
	insert(firstAnchor, "first tertiary ignorable", []int{0, 0, 0, 0})
	insert(lastAnchor, "last tertiary ignorable", []int{0, 0, 0, max})
	insert(lastAnchor, "last primary ignorable", []int{0, defaultSecondary, defaultTertiary, max})
	insert(lastAnchor, "last non ignorable", []int{maxPrimary, defaultSecondary, defaultTertiary, max})
	insert(lastAnchor, "__END__", []int{1 << maxPrimaryBits, defaultSecondary, defaultTertiary, max})
	return o
}

// patchForInsert eleminates entries from the list with more than one collation element.
// The next and prev fields of the eliminated entries still point to appropriate
// values in the newly created list.
// It requires that sort has been called.
func (o *ordering) patchForInsert() {
	for i := 0; i < len(o.ordered)-1; {
		e := o.ordered[i]
		lev := e.level
		n := e.next
		for ; n != nil && len(n.elems) > 1; n = n.next {
			if n.level < lev {
				lev = n.level
			}
			n.skipRemove = true
		}
		for ; o.ordered[i] != n; i++ {
			o.ordered[i].level = lev
			o.ordered[i].next = n
			o.ordered[i+1].prev = e
		}
	}
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
			logical:   e.logical,
		}
		oo.insert(ne)
	}
	oo.sort() // link all ordering.
	oo.patchForInsert()
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
func (o *ordering) genColElems(str string) []rawCE {
	elems := []rawCE{}
	for _, r := range []rune(str) {
		for _, ce := range o.find(string(r)).elems {
			if ce.w[0] != 0 || ce.w[1] != 0 || ce.w[2] != 0 {
				elems = append(elems, ce)
			}
		}
	}
	return elems
}
