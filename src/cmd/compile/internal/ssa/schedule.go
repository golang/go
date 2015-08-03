// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Schedule the Values in each Block.  After this phase returns, the
// order of b.Values matters and is the order in which those values
// will appear in the assembly output.  For now it generates a
// reasonable valid schedule using a priority queue.  TODO(khr):
// schedule smarter.
func schedule(f *Func) {
	// For each value, the number of times it is used in the block
	// by values that have not been scheduled yet.
	uses := make([]int, f.NumValues())

	// "priority" for a value
	score := make([]int, f.NumValues())

	// scheduling order.  We queue values in this list in reverse order.
	var order []*Value

	// priority queue of legally schedulable (0 unscheduled uses) values
	var priq [4][]*Value

	for _, b := range f.Blocks {
		// Compute uses.
		for _, v := range b.Values {
			if v.Op != OpPhi {
				// Note: if a value is used by a phi, it does not induce
				// a scheduling edge because that use is from the
				// previous iteration.
				for _, w := range v.Args {
					if w.Block == b {
						uses[w.ID]++
					}
				}
			}
		}
		// Compute score.  Larger numbers are scheduled closer to the end of the block.
		for _, v := range b.Values {
			switch {
			case v.Op == OpPhi:
				// We want all the phis first.
				score[v.ID] = 0
			case v.Type.IsMemory():
				// Schedule stores as late as possible.
				// This makes sure that loads do not get scheduled
				// after a following store (1-live-memory requirement).
				score[v.ID] = 2
			case v.Type.IsFlags():
				// Schedule flag register generation as late as possible.
				// This makes sure that we only have one live flags
				// value at a time.
				score[v.ID] = 2
			default:
				score[v.ID] = 1
			}
		}
		if b.Control != nil {
			// Force the control value to be scheduled at the end.
			score[b.Control.ID] = 3
			// TODO: some times control values are used by other values
			// in the block.  So the control value will not appear at
			// the very end.  Decide if this is a problem or not.
		}

		// Initialize priority queue with schedulable values.
		for i := range priq {
			priq[i] = priq[i][:0]
		}
		for _, v := range b.Values {
			if uses[v.ID] == 0 {
				s := score[v.ID]
				priq[s] = append(priq[s], v)
			}
		}

		// Schedule highest priority value, update use counts, repeat.
		order = order[:0]
		for {
			// Find highest priority schedulable value.
			var v *Value
			for i := len(priq) - 1; i >= 0; i-- {
				n := len(priq[i])
				if n == 0 {
					continue
				}
				v = priq[i][n-1]
				priq[i] = priq[i][:n-1]
				break
			}
			if v == nil {
				break
			}

			// Add it to the schedule.
			order = append(order, v)

			// Update use counts of arguments.
			for _, w := range v.Args {
				if w.Block != b {
					continue
				}
				uses[w.ID]--
				if uses[w.ID] == 0 {
					// All uses scheduled, w is now schedulable.
					s := score[w.ID]
					priq[s] = append(priq[s], w)
				}
			}
		}
		if len(order) != len(b.Values) {
			f.Fatalf("schedule does not include all values")
		}
		for i := 0; i < len(b.Values); i++ {
			b.Values[i] = order[len(b.Values)-1-i]
		}
	}

	f.scheduled = true
}
