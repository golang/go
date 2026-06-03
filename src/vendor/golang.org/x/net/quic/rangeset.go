// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// A rangeset is a set of int64s, stored as an ordered list of non-overlapping,
// non-empty ranges.
//
// Rangesets are efficient for small numbers of ranges,
// which is expected to be the common case.
type rangeset[T ~int64] []i64range[T]

type i64range[T ~int64] struct {
	start, end T // [start, end)
}

// size returns the size of the range.
func (r i64range[T]) size() T {
	return r.end - r.start
}

// contains reports whether v is in the range.
func (r i64range[T]) contains(v T) bool {
	return r.start <= v && v < r.end
}

// add adds [start, end) to the set, combining it with existing ranges if necessary.
func (s *rangeset[T]) add(start, end T) {
	if start == end {
		return
	}
	for i := range *s {
		r := &(*s)[i]
		if r.start > end {
			// The new range comes before range i.
			s.insertrange(i, start, end)
			return
		}
		if start > r.end {
			// The new range comes after range i.
			continue
		}
		// The new range is adjacent to or overlapping range i.
		if start < r.start {
			r.start = start
		}
		if end <= r.end {
			return
		}
		// Possibly coalesce subsequent ranges into range i.
		r.end = end
		j := i + 1
		for ; j < len(*s) && r.end >= (*s)[j].start; j++ {
			if e := (*s)[j].end; e > r.end {
				// Range j ends after the new range.
				r.end = e
			}
		}
		s.removeranges(i+1, j)
		return
	}
	*s = append(*s, i64range[T]{start, end})
}

// sub removes [start, end) from the set.
func (s *rangeset[T]) sub(start, end T) {
	removefrom, removeto := -1, -1
	for i := range *s {
		r := &(*s)[i]
		if end < r.start {
			break
		}
		if r.end < start {
			continue
		}
		switch {
		case start <= r.start && end >= r.end:
			// Remove the entire range.
			if removefrom == -1 {
				removefrom = i
			}
			removeto = i + 1
		case start <= r.start:
			// Remove a prefix.
			r.start = end
		case end >= r.end:
			// Remove a suffix.
			r.end = start
		default:
			// Remove the middle, leaving two new ranges.
			rend := r.end
			r.end = start
			s.insertrange(i+1, end, rend)
			return
		}
	}
	if removefrom != -1 {
		s.removeranges(removefrom, removeto)
	}
}

// contains reports whether s contains v.
func (s rangeset[T]) contains(v T) bool {
	for _, r := range s {
		if v >= r.end {
			continue
		}
		if r.start <= v {
			return true
		}
		return false
	}
	return false
}

// rangeContaining returns the range containing v, or the range [0,0) if v is not in s.
func (s rangeset[T]) rangeContaining(v T) i64range[T] {
	for _, r := range s {
		if v >= r.end {
			continue
		}
		if r.start <= v {
			return r
		}
		break
	}
	return i64range[T]{0, 0}
}

// min returns the minimum value in the set, or 0 if empty.
func (s rangeset[T]) min() T {
	if len(s) == 0 {
		return 0
	}
	return s[0].start
}

// max returns the maximum value in the set, or 0 if empty.
func (s rangeset[T]) max() T {
	if len(s) == 0 {
		return 0
	}
	return s[len(s)-1].end - 1
}

// end returns the end of the last range in the set, or 0 if empty.
func (s rangeset[T]) end() T {
	if len(s) == 0 {
		return 0
	}
	return s[len(s)-1].end
}

// numRanges returns the number of ranges in the rangeset.
func (s rangeset[T]) numRanges() int {
	return len(s)
}

// size returns the size of all ranges in the rangeset.
func (s rangeset[T]) size() (total T) {
	for _, r := range s {
		total += r.size()
	}
	return total
}

// isrange reports if the rangeset covers exactly the range [start, end).
func (s rangeset[T]) isrange(start, end T) bool {
	switch len(s) {
	case 0:
		return start == 0 && end == 0
	case 1:
		return s[0].start == start && s[0].end == end
	}
	return false
}

// removeranges removes ranges [i,j).
func (s *rangeset[T]) removeranges(i, j int) {
	if i == j {
		return
	}
	copy((*s)[i:], (*s)[j:])
	*s = (*s)[:len(*s)-(j-i)]
}

// insert adds a new range at index i.
func (s *rangeset[T]) insertrange(i int, start, end T) {
	*s = append(*s, i64range[T]{})
	copy((*s)[i+1:], (*s)[i:])
	(*s)[i] = i64range[T]{start, end}
}
