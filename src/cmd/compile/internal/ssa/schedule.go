// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

const (
	ScorePhi = iota // towards top of block
	ScoreVarDef
	ScoreMemory
	ScoreDefault
	ScoreFlags
	ScoreControl // towards bottom of block

	ScoreCount // not a real score
)

// Schedule the Values in each Block. After this phase returns, the
// order of b.Values matters and is the order in which those values
// will appear in the assembly output. For now it generates a
// reasonable valid schedule using a priority queue. TODO(khr):
// schedule smarter.
func schedule(f *Func) {
	// For each value, the number of times it is used in the block
	// by values that have not been scheduled yet.
	uses := make([]int, f.NumValues())

	// "priority" for a value
	score := make([]uint8, f.NumValues())

	// scheduling order. We queue values in this list in reverse order.
	var order []*Value

	// priority queue of legally schedulable (0 unscheduled uses) values
	var priq [ScoreCount][]*Value

	// maps mem values to the next live memory value
	nextMem := make([]*Value, f.NumValues())
	// additional pretend arguments for each Value. Used to enforce load/store ordering.
	additionalArgs := make([][]*Value, f.NumValues())

	for _, b := range f.Blocks {
		// Find store chain for block.
		// Store chains for different blocks overwrite each other, so
		// the calculated store chain is good only for this block.
		for _, v := range b.Values {
			if v.Op != OpPhi && v.Type.IsMemory() {
				for _, w := range v.Args {
					if w.Type.IsMemory() {
						nextMem[w.ID] = v
					}
				}
			}
		}

		// Compute uses.
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// If a value is used by a phi, it does not induce
				// a scheduling edge because that use is from the
				// previous iteration.
				continue
			}
			for _, w := range v.Args {
				if w.Block == b {
					uses[w.ID]++
				}
				// Any load must come before the following store.
				if v.Type.IsMemory() || !w.Type.IsMemory() {
					continue // not a load
				}
				s := nextMem[w.ID]
				if s == nil || s.Block != b {
					continue
				}
				additionalArgs[s.ID] = append(additionalArgs[s.ID], v)
				uses[v.ID]++
			}
		}
		// Compute score. Larger numbers are scheduled closer to the end of the block.
		for _, v := range b.Values {
			switch {
			case v.Op == OpAMD64LoweredGetClosurePtr:
				// We also score GetLoweredClosurePtr as early as possible to ensure that the
				// context register is not stomped. GetLoweredClosurePtr should only appear
				// in the entry block where there are no phi functions, so there is no
				// conflict or ambiguity here.
				if b != f.Entry {
					f.Fatalf("LoweredGetClosurePtr appeared outside of entry block, b=%s", b.String())
				}
				score[v.ID] = ScorePhi
			case v.Op == OpPhi:
				// We want all the phis first.
				score[v.ID] = ScorePhi
			case v.Op == OpVarDef:
				// We want all the vardefs next.
				score[v.ID] = ScoreVarDef
			case v.Type.IsMemory():
				// Schedule stores as early as possible. This tends to
				// reduce register pressure. It also helps make sure
				// VARDEF ops are scheduled before the corresponding LEA.
				score[v.ID] = ScoreMemory
			case v.Type.IsFlags():
				// Schedule flag register generation as late as possible.
				// This makes sure that we only have one live flags
				// value at a time.
				score[v.ID] = ScoreFlags
			default:
				score[v.ID] = ScoreDefault
			}
		}
		if b.Control != nil && b.Control.Op != OpPhi {
			// Force the control value to be scheduled at the end,
			// unless it is a phi value (which must be first).
			score[b.Control.ID] = ScoreControl

			// Schedule values dependent on the control value at the end.
			// This reduces the number of register spills. We don't find
			// all values that depend on the control, just values with a
			// direct dependency. This is cheaper and in testing there
			// was no difference in the number of spills.
			for _, v := range b.Values {
				if v.Op != OpPhi {
					for _, a := range v.Args {
						if a == b.Control {
							score[v.ID] = ScoreControl
						}
					}
				}
			}
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
			for _, w := range additionalArgs[v.ID] {
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
