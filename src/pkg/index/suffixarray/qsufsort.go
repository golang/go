// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This algorithm is based on "Faster Suffix Sorting"
//   by N. Jesper Larsson and Kunihiko Sadakane
// paper: http://www.larsson.dogma.net/ssrev-tr.pdf
// code:  http://www.larsson.dogma.net/qsufsort.c

// This algorithm computes the suffix array sa by computing its inverse.
// Consecutive groups of suffixes in sa are labeled as sorted groups or
// unsorted groups. For a given pass of the sorter, all suffixes are ordered
// up to their first h characters, and sa is h-ordered. Suffixes in their
// final positions and unambiguouly sorted in h-order are in a sorted group.
// Consecutive groups of suffixes with identical first h characters are an
// unsorted group. In each pass of the algorithm, unsorted groups are sorted
// according to the group number of their following suffix.

// In the implementation, if sa[i] is negative, it indicates that i is
// the first element of a sorted group of length -sa[i], and can be skipped.
// An unsorted group sa[i:k] is given the group number of the index of its
// last element, k-1. The group numbers are stored in the inverse slice (inv),
// and when all groups are sorted, this slice is the inverse suffix array.

package suffixarray

import "sort"

func qsufsort(data []byte) []int {
	// initial sorting by first byte of suffix
	sa := sortedByFirstByte(data)
	if len(sa) < 2 {
		return sa
	}
	// initialize the group lookup table
	// this becomes the inverse of the suffix array when all groups are sorted
	inv := initGroups(sa, data)

	// the index starts 1-ordered
	sufSortable := &suffixSortable{sa, inv, 1}

	for sa[0] > -len(sa) { // until all suffixes are one big sorted group
		// The suffixes are h-ordered, make them 2*h-ordered
		pi := 0 // pi is first position of first group
		sl := 0 // sl is negated length of sorted groups
		for pi < len(sa) {
			if s := sa[pi]; s < 0 { // if pi starts sorted group
				pi -= s // skip over sorted group
				sl += s // add negated length to sl
			} else { // if pi starts unsorted group
				if sl != 0 {
					sa[pi+sl] = sl // combine sorted groups before pi
					sl = 0
				}
				pk := inv[s] + 1 // pk-1 is last position of unsorted group
				sufSortable.sa = sa[pi:pk]
				sort.Sort(sufSortable)
				sufSortable.updateGroups(pi)
				pi = pk // next group
			}
		}
		if sl != 0 { // if the array ends with a sorted group
			sa[pi+sl] = sl // combine sorted groups at end of sa
		}

		sufSortable.h *= 2 // double sorted depth
	}

	for i := range sa { // reconstruct suffix array from inverse
		sa[inv[i]] = i
	}
	return sa
}


func sortedByFirstByte(data []byte) []int {
	// total byte counts
	var count [256]int
	for _, b := range data {
		count[b]++
	}
	// make count[b] equal index of first occurence of b in sorted array
	sum := 0
	for b := range count {
		count[b], sum = sum, count[b]+sum
	}
	// iterate through bytes, placing index into the correct spot in sa
	sa := make([]int, len(data))
	for i, b := range data {
		sa[count[b]] = i
		count[b]++
	}
	return sa
}


func initGroups(sa []int, data []byte) []int {
	// label contiguous same-letter groups with the same group number
	inv := make([]int, len(data))
	prevGroup := len(sa) - 1
	groupByte := data[sa[prevGroup]]
	for i := len(sa) - 1; i >= 0; i-- {
		if b := data[sa[i]]; b < groupByte {
			if prevGroup == i+1 {
				sa[i+1] = -1
			}
			groupByte = b
			prevGroup = i
		}
		inv[sa[i]] = prevGroup
		if prevGroup == 0 {
			sa[0] = -1
		}
	}
	// Separate out the final suffix to the start of its group.
	// This is necessary to ensure the suffix "a" is before "aba"
	// when using a potentially unstable sort.
	lastByte := data[len(data)-1]
	s := -1
	for i := range sa {
		if sa[i] >= 0 {
			if data[sa[i]] == lastByte && s == -1 {
				s = i
			}
			if sa[i] == len(sa)-1 {
				sa[i], sa[s] = sa[s], sa[i]
				inv[sa[s]] = s
				sa[s] = -1 // mark it as an isolated sorted group
				break
			}
		}
	}
	return inv
}


type suffixSortable struct {
	sa  []int
	inv []int
	h   int
}

func (x *suffixSortable) Len() int           { return len(x.sa) }
func (x *suffixSortable) Less(i, j int) bool { return x.inv[x.sa[i]+x.h] < x.inv[x.sa[j]+x.h] }
func (x *suffixSortable) Swap(i, j int)      { x.sa[i], x.sa[j] = x.sa[j], x.sa[i] }


func (x *suffixSortable) updateGroups(offset int) {
	prev := len(x.sa) - 1
	group := x.inv[x.sa[prev]+x.h]
	for i := prev; i >= 0; i-- {
		if g := x.inv[x.sa[i]+x.h]; g < group {
			if prev == i+1 { // previous group had size 1 and is thus sorted
				x.sa[i+1] = -1
			}
			group = g
			prev = i
		}
		x.inv[x.sa[i]] = prev + offset
		if prev == 0 { // first group has size 1 and is thus sorted
			x.sa[0] = -1
		}
	}
}
