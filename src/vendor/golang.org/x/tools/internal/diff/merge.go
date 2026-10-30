// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"slices"
)

// Merge merges two valid, ordered lists of edits.
// It returns zero if there was a conflict.
//
// If corresponding edits in x and y are identical,
// they are coalesced in the result.
//
// If x and y both provide different insertions at the same point,
// the insertions from x will be first in the result.
//
// TODO(adonovan): this algorithm could be improved, for example by
// working harder to coalesce non-identical edits that share a common
// deletion or common prefix of insertion (see the tests).
// Survey the academic literature for insights.
func Merge(x, y []Edit) ([]Edit, bool) {
	// Make a defensive (premature) copy of the arrays.
	x = slices.Clone(x)
	y = slices.Clone(y)

	var merged []Edit
	add := func(edit Edit) {
		merged = append(merged, edit)
	}
	var xi, yi int
	for xi < len(x) && yi < len(y) {
		px := &x[xi]
		py := &y[yi]

		if *px == *py {
			// x and y are identical: coalesce.
			add(*px)
			xi++
			yi++

		} else if px.End <= py.Start {
			// x is entirely before y,
			// or an insertion at start of y.
			add(*px)
			xi++

		} else if py.End <= px.Start {
			// y is entirely before x,
			// or an insertion at start of x.
			add(*py)
			yi++

		} else if px.Start < py.Start {
			// x is partly before y:
			// split it into a deletion and an edit.
			add(Edit{px.Start, py.Start, ""})
			px.Start = py.Start

		} else if py.Start < px.Start {
			// y is partly before x:
			// split it into a deletion and an edit.
			add(Edit{py.Start, px.Start, ""})
			py.Start = px.Start

		} else {
			// x and y are unequal non-insertions
			// at the same point: conflict.
			return nil, false
		}
	}
	for ; xi < len(x); xi++ {
		add(x[xi])
	}
	for ; yi < len(y); yi++ {
		add(y[yi])
	}
	return merged, true
}
