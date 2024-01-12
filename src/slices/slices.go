// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slices defines various functions useful with slices of any type.
package slices

import (
	"cmp"
	"unsafe"
)

// Equal reports whether two slices are equal: the same length and all
// elements equal. If the lengths are different, Equal returns false.
// Otherwise, the elements are compared in increasing index order, and the
// comparison stops at the first unequal pair.
// Floating point NaNs are not considered equal.
func Equal[S ~[]E, E comparable](s1, s2 S) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// EqualFunc reports whether two slices are equal using an equality
// function on each pair of elements. If the lengths are different,
// EqualFunc returns false. Otherwise, the elements are compared in
// increasing index order, and the comparison stops at the first index
// for which eq returns false.
func EqualFunc[S1 ~[]E1, S2 ~[]E2, E1, E2 any](s1 S1, s2 S2, eq func(E1, E2) bool) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if !eq(v1, v2) {
			return false
		}
	}
	return true
}

// Compare compares the elements of s1 and s2, using [cmp.Compare] on each pair
// of elements. The elements are compared sequentially, starting at index 0,
// until one element is not equal to the other.
// The result of comparing the first non-matching elements is returned.
// If both slices are equal until one of them ends, the shorter slice is
// considered less than the longer one.
// The result is 0 if s1 == s2, -1 if s1 < s2, and +1 if s1 > s2.
func Compare[S ~[]E, E cmp.Ordered](s1, s2 S) int {
	for i, v1 := range s1 {
		if i >= len(s2) {
			return +1
		}
		v2 := s2[i]
		if c := cmp.Compare(v1, v2); c != 0 {
			return c
		}
	}
	if len(s1) < len(s2) {
		return -1
	}
	return 0
}

// CompareFunc is like [Compare] but uses a custom comparison function on each
// pair of elements.
// The result is the first non-zero result of cmp; if cmp always
// returns 0 the result is 0 if len(s1) == len(s2), -1 if len(s1) < len(s2),
// and +1 if len(s1) > len(s2).
func CompareFunc[S1 ~[]E1, S2 ~[]E2, E1, E2 any](s1 S1, s2 S2, cmp func(E1, E2) int) int {
	for i, v1 := range s1 {
		if i >= len(s2) {
			return +1
		}
		v2 := s2[i]
		if c := cmp(v1, v2); c != 0 {
			return c
		}
	}
	if len(s1) < len(s2) {
		return -1
	}
	return 0
}

// Index returns the index of the first occurrence of v in s,
// or -1 if not present.
func Index[S ~[]E, E comparable](s S, v E) int {
	for i := range s {
		if v == s[i] {
			return i
		}
	}
	return -1
}

// IndexFunc returns the first index i satisfying f(s[i]),
// or -1 if none do.
func IndexFunc[S ~[]E, E any](s S, f func(E) bool) int {
	for i := range s {
		if f(s[i]) {
			return i
		}
	}
	return -1
}

// Contains reports whether v is present in s.
func Contains[S ~[]E, E comparable](s S, v E) bool {
	return Index(s, v) >= 0
}

// ContainsFunc reports whether at least one
// element e of s satisfies f(e).
func ContainsFunc[S ~[]E, E any](s S, f func(E) bool) bool {
	return IndexFunc(s, f) >= 0
}

// Insert inserts the values v... into s at index i,
// returning the modified slice.
// The elements at s[i:] are shifted up to make room.
// In the returned slice r, r[i] == v[0],
// and r[i+len(v)] == value originally at r[i].
// Insert panics if i is out of range.
// This function is O(len(s) + len(v)).
func Insert[S ~[]E, E any](s S, i int, v ...E) S {
	_ = s[i:] // bounds check

	m := len(v)
	if m == 0 {
		return s
	}
	n := len(s)
	if i == n {
		return append(s, v...)
	}
	if n+m > cap(s) {
		// Use append rather than make so that we bump the size of
		// the slice up to the next storage class.
		// This is what Grow does but we don't call Grow because
		// that might copy the values twice.
		s2 := append(s[:i], make(S, n+m-i)...)
		copy(s2[i:], v)
		copy(s2[i+m:], s[i:])
		return s2
	}
	s = s[:n+m]

	// before:
	// s: aaaaaaaabbbbccccccccdddd
	//            ^   ^       ^   ^
	//            i  i+m      n  n+m
	// after:
	// s: aaaaaaaavvvvbbbbcccccccc
	//            ^   ^       ^   ^
	//            i  i+m      n  n+m
	//
	// a are the values that don't move in s.
	// v are the values copied in from v.
	// b and c are the values from s that are shifted up in index.
	// d are the values that get overwritten, never to be seen again.

	if !overlaps(v, s[i+m:]) {
		// Easy case - v does not overlap either the c or d regions.
		// (It might be in some of a or b, or elsewhere entirely.)
		// The data we copy up doesn't write to v at all, so just do it.

		copy(s[i+m:], s[i:])

		// Now we have
		// s: aaaaaaaabbbbbbbbcccccccc
		//            ^   ^       ^   ^
		//            i  i+m      n  n+m
		// Note the b values are duplicated.

		copy(s[i:], v)

		// Now we have
		// s: aaaaaaaavvvvbbbbcccccccc
		//            ^   ^       ^   ^
		//            i  i+m      n  n+m
		// That's the result we want.
		return s
	}

	// The hard case - v overlaps c or d. We can't just shift up
	// the data because we'd move or clobber the values we're trying
	// to insert.
	// So instead, write v on top of d, then rotate.
	copy(s[n:], v)

	// Now we have
	// s: aaaaaaaabbbbccccccccvvvv
	//            ^   ^       ^   ^
	//            i  i+m      n  n+m

	rotateRight(s[i:], m)

	// Now we have
	// s: aaaaaaaavvvvbbbbcccccccc
	//            ^   ^       ^   ^
	//            i  i+m      n  n+m
	// That's the result we want.
	return s
}

// Delete removes the elements s[i:j] from s, returning the modified slice.
// Delete panics if j > len(s) or s[i:j] is not a valid slice of s.
// Delete is O(len(s)-i), so if many items must be deleted, it is better to
// make a single call deleting them all together than to delete one at a time.
// Delete zeroes the elements s[len(s)-(j-i):len(s)].
func Delete[S ~[]E, E any](s S, i, j int) S {
	_ = s[i:j:len(s)] // bounds check

	if i == j {
		return s
	}

	oldlen := len(s)
	s = append(s[:i], s[j:]...)
	clear(s[len(s):oldlen]) // zero/nil out the obsolete elements, for GC
	return s
}

// DeleteFunc removes any elements from s for which del returns true,
// returning the modified slice.
// DeleteFunc zeroes the elements between the new length and the original length.
func DeleteFunc[S ~[]E, E any](s S, del func(E) bool) S {
	i := IndexFunc(s, del)
	if i == -1 {
		return s
	}
	// Don't start copying elements until we find one to delete.
	for j := i + 1; j < len(s); j++ {
		if v := s[j]; !del(v) {
			s[i] = v
			i++
		}
	}
	clear(s[i:]) // zero/nil out the obsolete elements, for GC
	return s[:i]
}

// Replace replaces the elements s[i:j] by the given v, and returns the
// modified slice.
// Replace panics if j > len(s) or s[i:j] is not a valid slice of s.
// When len(v) < (j-i), Replace zeroes the elements between the new length and the original length.
func Replace[S ~[]E, E any](s S, i, j int, v ...E) S {
	_ = s[i:j] // bounds check

	if i == j {
		return Insert(s, i, v...)
	}
	if j == len(s) {
		return append(s[:i], v...)
	}

	tot := len(s[:i]) + len(v) + len(s[j:])
	if tot > cap(s) {
		// Too big to fit, allocate and copy over.
		s2 := append(s[:i], make(S, tot-i)...) // See Insert
		copy(s2[i:], v)
		copy(s2[i+len(v):], s[j:])
		return s2
	}

	r := s[:tot]

	if i+len(v) <= j {
		// Easy, as v fits in the deleted portion.
		copy(r[i:], v)
		copy(r[i+len(v):], s[j:])
		clear(s[tot:]) // zero/nil out the obsolete elements, for GC
		return r
	}

	// We are expanding (v is bigger than j-i).
	// The situation is something like this:
	// (example has i=4,j=8,len(s)=16,len(v)=6)
	// s: aaaaxxxxbbbbbbbbyy
	//        ^   ^       ^ ^
	//        i   j  len(s) tot
	// a: prefix of s
	// x: deleted range
	// b: more of s
	// y: area to expand into

	if !overlaps(r[i+len(v):], v) {
		// Easy, as v is not clobbered by the first copy.
		copy(r[i+len(v):], s[j:])
		copy(r[i:], v)
		return r
	}

	// This is a situation where we don't have a single place to which
	// we can copy v. Parts of it need to go to two different places.
	// We want to copy the prefix of v into y and the suffix into x, then
	// rotate |y| spots to the right.
	//
	//        v[2:]      v[:2]
	//         |           |
	// s: aaaavvvvbbbbbbbbvv
	//        ^   ^       ^ ^
	//        i   j  len(s) tot
	//
	// If either of those two destinations don't alias v, then we're good.
	y := len(v) - (j - i) // length of y portion

	if !overlaps(r[i:j], v) {
		copy(r[i:j], v[y:])
		copy(r[len(s):], v[:y])
		rotateRight(r[i:], y)
		return r
	}
	if !overlaps(r[len(s):], v) {
		copy(r[len(s):], v[:y])
		copy(r[i:j], v[y:])
		rotateRight(r[i:], y)
		return r
	}

	// Now we know that v overlaps both x and y.
	// That means that the entirety of b is *inside* v.
	// So we don't need to preserve b at all; instead we
	// can copy v first, then copy the b part of v out of
	// v to the right destination.
	k := startIdx(v, s[j:])
	copy(r[i:], v)
	copy(r[i+len(v):], r[i+k:])
	return r
}

// Clone returns a copy of the slice.
// The elements are copied using assignment, so this is a shallow clone.
func Clone[S ~[]E, E any](s S) S {
	if s == nil {
		return nil
	}
	clone := make(S, len(s))
	copy(clone, s)
	return clone
}

// Compact replaces consecutive runs of equal elements with a single copy.
// This is like the uniq command found on Unix.
// Compact modifies the contents of the slice s and returns the modified slice,
// which may have a smaller length.
// Compact zeroes the elements between the new length and the original length.
func Compact[S ~[]E, E comparable](s S) S {
	if len(s) < 2 {
		return s
	}
	i := 1
	for k := 1; k < len(s); k++ {
		if s[k] != s[k-1] {
			if i != k {
				s[i] = s[k]
			}
			i++
		}
	}
	clear(s[i:]) // zero/nil out the obsolete elements, for GC
	return s[:i]
}

// CompactFunc is like [Compact] but uses an equality function to compare elements.
// For runs of elements that compare equal, CompactFunc keeps the first one.
// CompactFunc zeroes the elements between the new length and the original length.
func CompactFunc[S ~[]E, E any](s S, eq func(E, E) bool) S {
	if len(s) < 2 {
		return s
	}
	i := 1
	for k := 1; k < len(s); k++ {
		if !eq(s[k], s[k-1]) {
			if i != k {
				s[i] = s[k]
			}
			i++
		}
	}
	clear(s[i:]) // zero/nil out the obsolete elements, for GC
	return s[:i]
}

// Grow increases the slice's capacity, if necessary, to guarantee space for
// another n elements. After Grow(n), at least n elements can be appended
// to the slice without another allocation. If n is negative or too large to
// allocate the memory, Grow panics.
func Grow[S ~[]E, E any](s S, n int) S {
	if n < 0 {
		panic("cannot be negative")
	}
	if n -= cap(s) - len(s); n > 0 {
		s = append(s[:cap(s)], make([]E, n)...)[:len(s)]
	}
	return s
}

// Clip removes unused capacity from the slice, returning s[:len(s):len(s)].
func Clip[S ~[]E, E any](s S) S {
	return s[:len(s):len(s)]
}

// Rotation algorithm explanation:
//
// rotate left by 2
// start with
//   0123456789
// split up like this
//   01 234567 89
// swap first 2 and last 2
//   89 234567 01
// join first parts
//   89234567 01
// recursively rotate first left part by 2
//   23456789 01
// join at the end
//   2345678901
//
// rotate left by 8
// start with
//   0123456789
// split up like this
//   01 234567 89
// swap first 2 and last 2
//   89 234567 01
// join last parts
//   89 23456701
// recursively rotate second part left by 6
//   89 01234567
// join at the end
//   8901234567

// TODO: There are other rotate algorithms.
// This algorithm has the desirable property that it moves each element exactly twice.
// The triple-reverse algorithm is simpler and more cache friendly, but takes more writes.
// The follow-cycles algorithm can be 1-write but it is not very cache friendly.

// rotateLeft rotates b left by n spaces.
// s_final[i] = s_orig[i+r], wrapping around.
func rotateLeft[E any](s []E, r int) {
	for r != 0 && r != len(s) {
		if r*2 <= len(s) {
			swap(s[:r], s[len(s)-r:])
			s = s[:len(s)-r]
		} else {
			swap(s[:len(s)-r], s[r:])
			s, r = s[len(s)-r:], r*2-len(s)
		}
	}
}
func rotateRight[E any](s []E, r int) {
	rotateLeft(s, len(s)-r)
}

// swap swaps the contents of x and y. x and y must be equal length and disjoint.
func swap[E any](x, y []E) {
	for i := 0; i < len(x); i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// overlaps reports whether the memory ranges a[0:len(a)] and b[0:len(b)] overlap.
func overlaps[E any](a, b []E) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	elemSize := unsafe.Sizeof(a[0])
	if elemSize == 0 {
		return false
	}
	// TODO: use a runtime/unsafe facility once one becomes available. See issue 12445.
	// Also see crypto/internal/alias/alias.go:AnyOverlap
	return uintptr(unsafe.Pointer(&a[0])) <= uintptr(unsafe.Pointer(&b[len(b)-1]))+(elemSize-1) &&
		uintptr(unsafe.Pointer(&b[0])) <= uintptr(unsafe.Pointer(&a[len(a)-1]))+(elemSize-1)
}

// startIdx returns the index in haystack where the needle starts.
// prerequisite: the needle must be aliased entirely inside the haystack.
func startIdx[E any](haystack, needle []E) int {
	p := &needle[0]
	for i := range haystack {
		if p == &haystack[i] {
			return i
		}
	}
	// TODO: what if the overlap is by a non-integral number of Es?
	panic("needle not found")
}

// Reverse reverses the elements of the slice in place.
func Reverse[S ~[]E, E any](s S) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

// Concat returns a new slice concatenating the passed in slices.
func Concat[S ~[]E, E any](slices ...S) S {
	size := 0
	for _, s := range slices {
		size += len(s)
		if size < 0 {
			panic("len out of range")
		}
	}
	newslice := Grow[S](nil, size)
	for _, s := range slices {
		newslice = append(newslice, s...)
	}
	return newslice
}
