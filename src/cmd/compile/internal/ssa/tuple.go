// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// tightenTupleSelectors ensures that tuple selectors (Select0, Select1,
// and SelectN ops) are in the same block as their tuple generator. The
// function also ensures that there are no duplicate tuple selectors.
// These properties are expected by the scheduler but may not have
// been maintained by the optimization pipeline up to this point.
//
// See issues 16741 and 39472.
func tightenTupleSelectors(f *Func) {
	selectors := make(map[struct {
		id    ID
		which int
	}]*Value)
	for _, b := range f.Blocks {
		for _, selector := range b.Values {
			// Key fields for de-duplication
			var tuple *Value
			idx := 0
			switch selector.Op {
			default:
				continue
			case OpSelect1:
				idx = 1
				fallthrough
			case OpSelect0:
				tuple = selector.Args[0]
				if !tuple.Type.IsTuple() {
					f.Fatalf("arg of tuple selector %s is not a tuple: %s", selector.String(), tuple.LongString())
				}
			case OpSelectN:
				tuple = selector.Args[0]
				idx = int(selector.AuxInt)
				if !tuple.Type.IsResults() {
					f.Fatalf("arg of result selector %s is not a results: %s", selector.String(), tuple.LongString())
				}
			}

			// If there is a pre-existing selector in the target block then
			// use that. Do this even if the selector is already in the
			// target block to avoid duplicate tuple selectors.
			key := struct {
				id    ID
				which int
			}{tuple.ID, idx}
			if t := selectors[key]; t != nil {
				if selector != t {
					selector.copyOf(t)
				}
				continue
			}

			// If the selector is in the wrong block copy it into the target
			// block.
			if selector.Block != tuple.Block {
				t := selector.copyInto(tuple.Block)
				selector.copyOf(t)
				selectors[key] = t
				continue
			}

			// The selector is in the target block. Add it to the map so it
			// cannot be duplicated.
			selectors[key] = selector
		}
	}
}
