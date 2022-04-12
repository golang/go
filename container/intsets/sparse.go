// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package intsets provides Sparse, a compact and fast representation
// for sparse sets of int values.
//
// The time complexity of the operations Len, Insert, Remove and Has
// is in O(n) but in practice those methods are faster and more
// space-efficient than equivalent operations on sets based on the Go
// map type.  The IsEmpty, Min, Max, Clear and TakeMin operations
// require constant time.
package intsets // import "golang.org/x/tools/container/intsets"

// TODO(adonovan):
// - Add InsertAll(...int), RemoveAll(...int)
// - Add 'bool changed' results for {Intersection,Difference}With too.
//
// TODO(adonovan): implement Dense, a dense bit vector with a similar API.
// The space usage would be proportional to Max(), not Len(), and the
// implementation would be based upon big.Int.
//
// TODO(adonovan): opt: make UnionWith and Difference faster.
// These are the hot-spots for go/pointer.

import (
	"bytes"
	"fmt"
	"math/bits"
)

// A Sparse is a set of int values.
// Sparse operations (even queries) are not concurrency-safe.
//
// The zero value for Sparse is a valid empty set.
//
// Sparse sets must be copied using the Copy method, not by assigning
// a Sparse value.
type Sparse struct {
	// An uninitialized Sparse represents an empty set.
	// An empty set may also be represented by
	//  root.next == root.prev == &root.
	//
	// The root is always the block with the smallest offset.
	// It can be empty, but only if it is the only block; in that case, offset is
	// MaxInt (which is not a valid offset).
	root block
}

type word uintptr

const (
	_m            = ^word(0)
	bitsPerWord   = 8 << (_m>>8&1 + _m>>16&1 + _m>>32&1)
	bitsPerBlock  = 256 // optimal value for go/pointer solver performance
	wordsPerBlock = bitsPerBlock / bitsPerWord
)

// Limit values of implementation-specific int type.
const (
	MaxInt = int(^uint(0) >> 1)
	MinInt = -MaxInt - 1
)

// popcount returns the number of set bits in w.
func popcount(x word) int {
	// Avoid OnesCount(uint): don't assume uint = uintptr.
	if bitsPerWord == 32 {
		return bits.OnesCount32(uint32(x))
	} else {
		return bits.OnesCount64(uint64(x))
	}
}

// nlz returns the number of leading zeros of x.
func nlz(x word) int {
	// Avoid LeadingZeros(uint): don't assume uint = uintptr.
	if bitsPerWord == 32 {
		return bits.LeadingZeros32(uint32(x))
	} else {
		return bits.LeadingZeros64(uint64(x))
	}
}

// ntz returns the number of trailing zeros of x.
func ntz(x word) int {
	// Avoid TrailingZeros(uint): don't assume uint = uintptr.
	if bitsPerWord == 32 {
		return bits.TrailingZeros32(uint32(x))
	} else {
		return bits.TrailingZeros64(uint64(x))
	}
}

// -- block ------------------------------------------------------------

// A set is represented as a circular doubly-linked list of blocks,
// each containing an offset and a bit array of fixed size
// bitsPerBlock; the blocks are ordered by increasing offset.
//
// The set contains an element x iff the block whose offset is x - (x
// mod bitsPerBlock) has the bit (x mod bitsPerBlock) set, where mod
// is the Euclidean remainder.
//
// A block may only be empty transiently.
type block struct {
	offset     int                 // offset mod bitsPerBlock == 0
	bits       [wordsPerBlock]word // contains at least one set bit
	next, prev *block              // doubly-linked list of blocks
}

// wordMask returns the word index (in block.bits)
// and single-bit mask for the block's ith bit.
func wordMask(i uint) (w uint, mask word) {
	w = i / bitsPerWord
	mask = 1 << (i % bitsPerWord)
	return
}

// insert sets the block b's ith bit and
// returns true if it was not already set.
func (b *block) insert(i uint) bool {
	w, mask := wordMask(i)
	if b.bits[w]&mask == 0 {
		b.bits[w] |= mask
		return true
	}
	return false
}

// remove clears the block's ith bit and
// returns true if the bit was previously set.
// NB: may leave the block empty.
func (b *block) remove(i uint) bool {
	w, mask := wordMask(i)
	if b.bits[w]&mask != 0 {
		b.bits[w] &^= mask
		return true
	}
	return false
}

// has reports whether the block's ith bit is set.
func (b *block) has(i uint) bool {
	w, mask := wordMask(i)
	return b.bits[w]&mask != 0
}

// empty reports whether b.len()==0, but more efficiently.
func (b *block) empty() bool {
	for _, w := range b.bits {
		if w != 0 {
			return false
		}
	}
	return true
}

// len returns the number of set bits in block b.
func (b *block) len() int {
	var l int
	for _, w := range b.bits {
		l += popcount(w)
	}
	return l
}

// max returns the maximum element of the block.
// The block must not be empty.
func (b *block) max() int {
	bi := b.offset + bitsPerBlock
	// Decrement bi by number of high zeros in last.bits.
	for i := len(b.bits) - 1; i >= 0; i-- {
		if w := b.bits[i]; w != 0 {
			return bi - nlz(w) - 1
		}
		bi -= bitsPerWord
	}
	panic("BUG: empty block")
}

// min returns the minimum element of the block,
// and also removes it if take is set.
// The block must not be initially empty.
// NB: may leave the block empty.
func (b *block) min(take bool) int {
	for i, w := range b.bits {
		if w != 0 {
			tz := ntz(w)
			if take {
				b.bits[i] = w &^ (1 << uint(tz))
			}
			return b.offset + int(i*bitsPerWord) + tz
		}
	}
	panic("BUG: empty block")
}

// lowerBound returns the smallest element of the block that is greater than or
// equal to the element corresponding to the ith bit. If there is no such
// element, the second return value is false.
func (b *block) lowerBound(i uint) (int, bool) {
	w := i / bitsPerWord
	bit := i % bitsPerWord

	if val := b.bits[w] >> bit; val != 0 {
		return b.offset + int(i) + ntz(val), true
	}

	for w++; w < wordsPerBlock; w++ {
		if val := b.bits[w]; val != 0 {
			return b.offset + int(w*bitsPerWord) + ntz(val), true
		}
	}

	return 0, false
}

// forEach calls f for each element of block b.
// f must not mutate b's enclosing Sparse.
func (b *block) forEach(f func(int)) {
	for i, w := range b.bits {
		offset := b.offset + i*bitsPerWord
		for bi := 0; w != 0 && bi < bitsPerWord; bi++ {
			if w&1 != 0 {
				f(offset)
			}
			offset++
			w >>= 1
		}
	}
}

// offsetAndBitIndex returns the offset of the block that would
// contain x and the bit index of x within that block.
func offsetAndBitIndex(x int) (int, uint) {
	mod := x % bitsPerBlock
	if mod < 0 {
		// Euclidean (non-negative) remainder
		mod += bitsPerBlock
	}
	return x - mod, uint(mod)
}

// -- Sparse --------------------------------------------------------------

// none is a shared, empty, sentinel block that indicates the end of a block
// list.
var none block

// Dummy type used to generate an implicit panic. This must be defined at the
// package level; if it is defined inside a function, it prevents the inlining
// of that function.
type to_copy_a_sparse_you_must_call_its_Copy_method struct{}

// init ensures s is properly initialized.
func (s *Sparse) init() {
	root := &s.root
	if root.next == nil {
		root.offset = MaxInt
		root.next = root
		root.prev = root
	} else if root.next.prev != root {
		// Copying a Sparse x leads to pernicious corruption: the
		// new Sparse y shares the old linked list, but iteration
		// on y will never encounter &y.root so it goes into a
		// loop.  Fail fast before this occurs.
		// We don't want to call panic here because it prevents the
		// inlining of this function.
		_ = (interface{}(nil)).(to_copy_a_sparse_you_must_call_its_Copy_method)
	}
}

func (s *Sparse) first() *block {
	s.init()
	if s.root.offset == MaxInt {
		return &none
	}
	return &s.root
}

// next returns the next block in the list, or end if b is the last block.
func (s *Sparse) next(b *block) *block {
	if b.next == &s.root {
		return &none
	}
	return b.next
}

// prev returns the previous block in the list, or end if b is the first block.
func (s *Sparse) prev(b *block) *block {
	if b.prev == &s.root {
		return &none
	}
	return b.prev
}

// IsEmpty reports whether the set s is empty.
func (s *Sparse) IsEmpty() bool {
	return s.root.next == nil || s.root.offset == MaxInt
}

// Len returns the number of elements in the set s.
func (s *Sparse) Len() int {
	var l int
	for b := s.first(); b != &none; b = s.next(b) {
		l += b.len()
	}
	return l
}

// Max returns the maximum element of the set s, or MinInt if s is empty.
func (s *Sparse) Max() int {
	if s.IsEmpty() {
		return MinInt
	}
	return s.root.prev.max()
}

// Min returns the minimum element of the set s, or MaxInt if s is empty.
func (s *Sparse) Min() int {
	if s.IsEmpty() {
		return MaxInt
	}
	return s.root.min(false)
}

// LowerBound returns the smallest element >= x, or MaxInt if there is no such
// element.
func (s *Sparse) LowerBound(x int) int {
	offset, i := offsetAndBitIndex(x)
	for b := s.first(); b != &none; b = s.next(b) {
		if b.offset > offset {
			return b.min(false)
		}
		if b.offset == offset {
			if y, ok := b.lowerBound(i); ok {
				return y
			}
		}
	}
	return MaxInt
}

// block returns the block that would contain offset,
// or nil if s contains no such block.
// Precondition: offset is a multiple of bitsPerBlock.
func (s *Sparse) block(offset int) *block {
	for b := s.first(); b != &none && b.offset <= offset; b = s.next(b) {
		if b.offset == offset {
			return b
		}
	}
	return nil
}

// Insert adds x to the set s, and reports whether the set grew.
func (s *Sparse) Insert(x int) bool {
	offset, i := offsetAndBitIndex(x)

	b := s.first()
	for ; b != &none && b.offset <= offset; b = s.next(b) {
		if b.offset == offset {
			return b.insert(i)
		}
	}

	// Insert new block before b.
	new := s.insertBlockBefore(b)
	new.offset = offset
	return new.insert(i)
}

// removeBlock removes a block and returns the block that followed it (or end if
// it was the last block).
func (s *Sparse) removeBlock(b *block) *block {
	if b != &s.root {
		b.prev.next = b.next
		b.next.prev = b.prev
		if b.next == &s.root {
			return &none
		}
		return b.next
	}

	first := s.root.next
	if first == &s.root {
		// This was the only block.
		s.Clear()
		return &none
	}
	s.root.offset = first.offset
	s.root.bits = first.bits
	if first.next == &s.root {
		// Single block remaining.
		s.root.next = &s.root
		s.root.prev = &s.root
	} else {
		s.root.next = first.next
		first.next.prev = &s.root
	}
	return &s.root
}

// Remove removes x from the set s, and reports whether the set shrank.
func (s *Sparse) Remove(x int) bool {
	offset, i := offsetAndBitIndex(x)
	if b := s.block(offset); b != nil {
		if !b.remove(i) {
			return false
		}
		if b.empty() {
			s.removeBlock(b)
		}
		return true
	}
	return false
}

// Clear removes all elements from the set s.
func (s *Sparse) Clear() {
	s.root = block{
		offset: MaxInt,
		next:   &s.root,
		prev:   &s.root,
	}
}

// If set s is non-empty, TakeMin sets *p to the minimum element of
// the set s, removes that element from the set and returns true.
// Otherwise, it returns false and *p is undefined.
//
// This method may be used for iteration over a worklist like so:
//
//	var x int
//	for worklist.TakeMin(&x) { use(x) }
func (s *Sparse) TakeMin(p *int) bool {
	if s.IsEmpty() {
		return false
	}
	*p = s.root.min(true)
	if s.root.empty() {
		s.removeBlock(&s.root)
	}
	return true
}

// Has reports whether x is an element of the set s.
func (s *Sparse) Has(x int) bool {
	offset, i := offsetAndBitIndex(x)
	if b := s.block(offset); b != nil {
		return b.has(i)
	}
	return false
}

// forEach applies function f to each element of the set s in order.
//
// f must not mutate s.  Consequently, forEach is not safe to expose
// to clients.  In any case, using "range s.AppendTo()" allows more
// natural control flow with continue/break/return.
func (s *Sparse) forEach(f func(int)) {
	for b := s.first(); b != &none; b = s.next(b) {
		b.forEach(f)
	}
}

// Copy sets s to the value of x.
func (s *Sparse) Copy(x *Sparse) {
	if s == x {
		return
	}

	xb := x.first()
	sb := s.first()
	for xb != &none {
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = xb.offset
		sb.bits = xb.bits
		xb = x.next(xb)
		sb = s.next(sb)
	}
	s.discardTail(sb)
}

// insertBlockBefore returns a new block, inserting it before next.
// If next is the root, the root is replaced. If next is end, the block is
// inserted at the end.
func (s *Sparse) insertBlockBefore(next *block) *block {
	if s.IsEmpty() {
		if next != &none {
			panic("BUG: passed block with empty set")
		}
		return &s.root
	}

	if next == &s.root {
		// Special case: we need to create a new block that will become the root
		// block.The old root block becomes the second block.
		second := s.root
		s.root = block{
			next: &second,
		}
		if second.next == &s.root {
			s.root.prev = &second
		} else {
			s.root.prev = second.prev
			second.next.prev = &second
			second.prev = &s.root
		}
		return &s.root
	}
	if next == &none {
		// Insert before root.
		next = &s.root
	}
	b := new(block)
	b.next = next
	b.prev = next.prev
	b.prev.next = b
	next.prev = b
	return b
}

// discardTail removes block b and all its successors from s.
func (s *Sparse) discardTail(b *block) {
	if b != &none {
		if b == &s.root {
			s.Clear()
		} else {
			b.prev.next = &s.root
			s.root.prev = b.prev
		}
	}
}

// IntersectionWith sets s to the intersection s ∩ x.
func (s *Sparse) IntersectionWith(x *Sparse) {
	if s == x {
		return
	}

	xb := x.first()
	sb := s.first()
	for xb != &none && sb != &none {
		switch {
		case xb.offset < sb.offset:
			xb = x.next(xb)

		case xb.offset > sb.offset:
			sb = s.removeBlock(sb)

		default:
			var sum word
			for i := range sb.bits {
				r := xb.bits[i] & sb.bits[i]
				sb.bits[i] = r
				sum |= r
			}
			if sum != 0 {
				sb = s.next(sb)
			} else {
				// sb will be overwritten or removed
			}

			xb = x.next(xb)
		}
	}

	s.discardTail(sb)
}

// Intersection sets s to the intersection x ∩ y.
func (s *Sparse) Intersection(x, y *Sparse) {
	switch {
	case s == x:
		s.IntersectionWith(y)
		return
	case s == y:
		s.IntersectionWith(x)
		return
	case x == y:
		s.Copy(x)
		return
	}

	xb := x.first()
	yb := y.first()
	sb := s.first()
	for xb != &none && yb != &none {
		switch {
		case xb.offset < yb.offset:
			xb = x.next(xb)
			continue
		case xb.offset > yb.offset:
			yb = y.next(yb)
			continue
		}

		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = xb.offset

		var sum word
		for i := range sb.bits {
			r := xb.bits[i] & yb.bits[i]
			sb.bits[i] = r
			sum |= r
		}
		if sum != 0 {
			sb = s.next(sb)
		} else {
			// sb will be overwritten or removed
		}

		xb = x.next(xb)
		yb = y.next(yb)
	}

	s.discardTail(sb)
}

// Intersects reports whether s ∩ x ≠ ∅.
func (s *Sparse) Intersects(x *Sparse) bool {
	sb := s.first()
	xb := x.first()
	for sb != &none && xb != &none {
		switch {
		case xb.offset < sb.offset:
			xb = x.next(xb)
		case xb.offset > sb.offset:
			sb = s.next(sb)
		default:
			for i := range sb.bits {
				if sb.bits[i]&xb.bits[i] != 0 {
					return true
				}
			}
			sb = s.next(sb)
			xb = x.next(xb)
		}
	}
	return false
}

// UnionWith sets s to the union s ∪ x, and reports whether s grew.
func (s *Sparse) UnionWith(x *Sparse) bool {
	if s == x {
		return false
	}

	var changed bool
	xb := x.first()
	sb := s.first()
	for xb != &none {
		if sb != &none && sb.offset == xb.offset {
			for i := range xb.bits {
				union := sb.bits[i] | xb.bits[i]
				if sb.bits[i] != union {
					sb.bits[i] = union
					changed = true
				}
			}
			xb = x.next(xb)
		} else if sb == &none || sb.offset > xb.offset {
			sb = s.insertBlockBefore(sb)
			sb.offset = xb.offset
			sb.bits = xb.bits
			changed = true

			xb = x.next(xb)
		}
		sb = s.next(sb)
	}
	return changed
}

// Union sets s to the union x ∪ y.
func (s *Sparse) Union(x, y *Sparse) {
	switch {
	case x == y:
		s.Copy(x)
		return
	case s == x:
		s.UnionWith(y)
		return
	case s == y:
		s.UnionWith(x)
		return
	}

	xb := x.first()
	yb := y.first()
	sb := s.first()
	for xb != &none || yb != &none {
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		switch {
		case yb == &none || (xb != &none && xb.offset < yb.offset):
			sb.offset = xb.offset
			sb.bits = xb.bits
			xb = x.next(xb)

		case xb == &none || (yb != &none && yb.offset < xb.offset):
			sb.offset = yb.offset
			sb.bits = yb.bits
			yb = y.next(yb)

		default:
			sb.offset = xb.offset
			for i := range xb.bits {
				sb.bits[i] = xb.bits[i] | yb.bits[i]
			}
			xb = x.next(xb)
			yb = y.next(yb)
		}
		sb = s.next(sb)
	}

	s.discardTail(sb)
}

// DifferenceWith sets s to the difference s ∖ x.
func (s *Sparse) DifferenceWith(x *Sparse) {
	if s == x {
		s.Clear()
		return
	}

	xb := x.first()
	sb := s.first()
	for xb != &none && sb != &none {
		switch {
		case xb.offset > sb.offset:
			sb = s.next(sb)

		case xb.offset < sb.offset:
			xb = x.next(xb)

		default:
			var sum word
			for i := range sb.bits {
				r := sb.bits[i] & ^xb.bits[i]
				sb.bits[i] = r
				sum |= r
			}
			if sum == 0 {
				sb = s.removeBlock(sb)
			} else {
				sb = s.next(sb)
			}
			xb = x.next(xb)
		}
	}
}

// Difference sets s to the difference x ∖ y.
func (s *Sparse) Difference(x, y *Sparse) {
	switch {
	case x == y:
		s.Clear()
		return
	case s == x:
		s.DifferenceWith(y)
		return
	case s == y:
		var y2 Sparse
		y2.Copy(y)
		s.Difference(x, &y2)
		return
	}

	xb := x.first()
	yb := y.first()
	sb := s.first()
	for xb != &none && yb != &none {
		if xb.offset > yb.offset {
			// y has block, x has &none
			yb = y.next(yb)
			continue
		}

		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = xb.offset

		switch {
		case xb.offset < yb.offset:
			// x has block, y has &none
			sb.bits = xb.bits

			sb = s.next(sb)

		default:
			// x and y have corresponding blocks
			var sum word
			for i := range sb.bits {
				r := xb.bits[i] & ^yb.bits[i]
				sb.bits[i] = r
				sum |= r
			}
			if sum != 0 {
				sb = s.next(sb)
			} else {
				// sb will be overwritten or removed
			}

			yb = y.next(yb)
		}
		xb = x.next(xb)
	}

	for xb != &none {
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = xb.offset
		sb.bits = xb.bits
		sb = s.next(sb)

		xb = x.next(xb)
	}

	s.discardTail(sb)
}

// SymmetricDifferenceWith sets s to the symmetric difference s ∆ x.
func (s *Sparse) SymmetricDifferenceWith(x *Sparse) {
	if s == x {
		s.Clear()
		return
	}

	sb := s.first()
	xb := x.first()
	for xb != &none && sb != &none {
		switch {
		case sb.offset < xb.offset:
			sb = s.next(sb)
		case xb.offset < sb.offset:
			nb := s.insertBlockBefore(sb)
			nb.offset = xb.offset
			nb.bits = xb.bits
			xb = x.next(xb)
		default:
			var sum word
			for i := range sb.bits {
				r := sb.bits[i] ^ xb.bits[i]
				sb.bits[i] = r
				sum |= r
			}
			if sum == 0 {
				sb = s.removeBlock(sb)
			} else {
				sb = s.next(sb)
			}
			xb = x.next(xb)
		}
	}

	for xb != &none { // append the tail of x to s
		sb = s.insertBlockBefore(sb)
		sb.offset = xb.offset
		sb.bits = xb.bits
		sb = s.next(sb)
		xb = x.next(xb)
	}
}

// SymmetricDifference sets s to the symmetric difference x ∆ y.
func (s *Sparse) SymmetricDifference(x, y *Sparse) {
	switch {
	case x == y:
		s.Clear()
		return
	case s == x:
		s.SymmetricDifferenceWith(y)
		return
	case s == y:
		s.SymmetricDifferenceWith(x)
		return
	}

	sb := s.first()
	xb := x.first()
	yb := y.first()
	for xb != &none && yb != &none {
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		switch {
		case yb.offset < xb.offset:
			sb.offset = yb.offset
			sb.bits = yb.bits
			sb = s.next(sb)
			yb = y.next(yb)
		case xb.offset < yb.offset:
			sb.offset = xb.offset
			sb.bits = xb.bits
			sb = s.next(sb)
			xb = x.next(xb)
		default:
			var sum word
			for i := range sb.bits {
				r := xb.bits[i] ^ yb.bits[i]
				sb.bits[i] = r
				sum |= r
			}
			if sum != 0 {
				sb.offset = xb.offset
				sb = s.next(sb)
			}
			xb = x.next(xb)
			yb = y.next(yb)
		}
	}

	for xb != &none { // append the tail of x to s
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = xb.offset
		sb.bits = xb.bits
		sb = s.next(sb)
		xb = x.next(xb)
	}

	for yb != &none { // append the tail of y to s
		if sb == &none {
			sb = s.insertBlockBefore(sb)
		}
		sb.offset = yb.offset
		sb.bits = yb.bits
		sb = s.next(sb)
		yb = y.next(yb)
	}

	s.discardTail(sb)
}

// SubsetOf reports whether s ∖ x = ∅.
func (s *Sparse) SubsetOf(x *Sparse) bool {
	if s == x {
		return true
	}

	sb := s.first()
	xb := x.first()
	for sb != &none {
		switch {
		case xb == &none || xb.offset > sb.offset:
			return false
		case xb.offset < sb.offset:
			xb = x.next(xb)
		default:
			for i := range sb.bits {
				if sb.bits[i]&^xb.bits[i] != 0 {
					return false
				}
			}
			sb = s.next(sb)
			xb = x.next(xb)
		}
	}
	return true
}

// Equals reports whether the sets s and t have the same elements.
func (s *Sparse) Equals(t *Sparse) bool {
	if s == t {
		return true
	}
	sb := s.first()
	tb := t.first()
	for {
		switch {
		case sb == &none && tb == &none:
			return true
		case sb == &none || tb == &none:
			return false
		case sb.offset != tb.offset:
			return false
		case sb.bits != tb.bits:
			return false
		}

		sb = s.next(sb)
		tb = t.next(tb)
	}
}

// String returns a human-readable description of the set s.
func (s *Sparse) String() string {
	var buf bytes.Buffer
	buf.WriteByte('{')
	s.forEach(func(x int) {
		if buf.Len() > 1 {
			buf.WriteByte(' ')
		}
		fmt.Fprintf(&buf, "%d", x)
	})
	buf.WriteByte('}')
	return buf.String()
}

// BitString returns the set as a string of 1s and 0s denoting the sum
// of the i'th powers of 2, for each i in s.  A radix point, always
// preceded by a digit, appears if the sum is non-integral.
//
// Examples:
//
//	        {}.BitString() =      "0"
//	     {4,5}.BitString() = "110000"
//	      {-3}.BitString() =      "0.001"
//	{-3,0,4,5}.BitString() = "110001.001"
func (s *Sparse) BitString() string {
	if s.IsEmpty() {
		return "0"
	}

	min, max := s.Min(), s.Max()
	var nbytes int
	if max > 0 {
		nbytes = max
	}
	nbytes++ // zero bit
	radix := nbytes
	if min < 0 {
		nbytes += len(".") - min
	}

	b := make([]byte, nbytes)
	for i := range b {
		b[i] = '0'
	}
	if radix < nbytes {
		b[radix] = '.'
	}
	s.forEach(func(x int) {
		if x >= 0 {
			x += len(".")
		}
		b[radix-x] = '1'
	})
	return string(b)
}

// GoString returns a string showing the internal representation of
// the set s.
func (s *Sparse) GoString() string {
	var buf bytes.Buffer
	for b := s.first(); b != &none; b = s.next(b) {
		fmt.Fprintf(&buf, "block %p {offset=%d next=%p prev=%p",
			b, b.offset, b.next, b.prev)
		for _, w := range b.bits {
			fmt.Fprintf(&buf, " 0%016x", w)
		}
		fmt.Fprintf(&buf, "}\n")
	}
	return buf.String()
}

// AppendTo returns the result of appending the elements of s to slice
// in order.
func (s *Sparse) AppendTo(slice []int) []int {
	s.forEach(func(x int) {
		slice = append(slice, x)
	})
	return slice
}

// -- Testing/debugging ------------------------------------------------

// check returns an error if the representation invariants of s are violated.
func (s *Sparse) check() error {
	s.init()
	if s.root.empty() {
		// An empty set must have only the root block with offset MaxInt.
		if s.root.next != &s.root {
			return fmt.Errorf("multiple blocks with empty root block")
		}
		if s.root.offset != MaxInt {
			return fmt.Errorf("empty set has offset %d, should be MaxInt", s.root.offset)
		}
		return nil
	}
	for b := s.first(); ; b = s.next(b) {
		if b.offset%bitsPerBlock != 0 {
			return fmt.Errorf("bad offset modulo: %d", b.offset)
		}
		if b.empty() {
			return fmt.Errorf("empty block")
		}
		if b.prev.next != b {
			return fmt.Errorf("bad prev.next link")
		}
		if b.next.prev != b {
			return fmt.Errorf("bad next.prev link")
		}
		if b.next == &s.root {
			break
		}
		if b.offset >= b.next.offset {
			return fmt.Errorf("bad offset order: b.offset=%d, b.next.offset=%d",
				b.offset, b.next.offset)
		}
	}
	return nil
}
