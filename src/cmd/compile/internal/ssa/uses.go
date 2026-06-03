// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// useInfo provides a map from a value to the users of that value.
// We do not keep track of this data in the IR directly, as it is
// expensive to keep updated. But sometimes we need to compute it
// temporarily.
//
// A useInfo is only valid until the next modification of the IR.
//
// Note that block control uses are not reported. (TODO: add
// if needed.)
// Also, index of use is not reported. (TODO: add if needed.)
//
// We keep track of all uses in a single array, and use a
// starts array to tell us where to find the sub-array for
// each value.
//
// For a function with 10 values, we would have:
//
//	starts[0]    starts[1]  starts[2] starts[9]   len(uses)
//	    |            |            |     |            |
//	    v            v            v     v            v
//	    +------------+------------+-...-+------------+
//	    | uses of v0 | uses of v1 | ... | uses of v9 |
//	    +------------+------------+-...-+------------+
//
// We can find all the uses of v by listing all entries
// of uses between starts[v.ID] and starts[v.ID+1].
type useInfo struct {
	starts []int32
	uses   []*Value
}

// build useInfo for a function. Result only valid until
// the next modification of f.
func uses(f *Func) useInfo {
	// Write down number of uses of each value.
	idx := f.Cache.allocInt32Slice(f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			idx[v.ID] = v.Uses
		}
	}

	// Compute cumulative sum of uses up to and
	// including each value ID.
	var cum int32
	for vid, uses := range idx {
		cum += uses
		idx[vid] = cum
	}

	// Compute uses.
	uses := f.Cache.allocValueSlice(int(cum))
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				idx[a.ID]--
				uses[idx[a.ID]] = v
			}
		}
	}
	for _, b := range f.Blocks {
		for _, c := range b.ControlValues() {
			// We don't track block control uses, but
			// we have to decrement idx values here
			// so that the accounting comes out right.
			// Each value will have, at the start of its
			// use list, a bunch of nils that represent
			// the number of Block.Control uses.
			idx[c.ID]--
		}
	}

	// The loop above decremented each idx entry
	// by the number of uses. It now contains
	// the sum of uses up to, but not including,
	// each value ID.
	return useInfo{starts: idx, uses: uses}
}

// get returns a list of uses of v.
// Every use in an argument slot is listed (e.g. for
// w=(Add v v), w is listed twice in the uses of v).
// Uses by Block.Controls are not reported.
func (u useInfo) get(v *Value) []*Value {
	i := u.starts[v.ID]
	var j int32
	if int(v.ID) < len(u.starts)-1 {
		j = u.starts[v.ID+1]
	} else {
		j = int32(len(u.uses))
	}
	r := u.uses[i:j]
	// skip nil entries from block control uses
	for len(r) > 0 && r[0] == nil {
		r = r[1:]
	}
	return r
}

func (u useInfo) free(f *Func) {
	f.Cache.freeInt32Slice(u.starts)
	f.Cache.freeValueSlice(u.uses)
}
