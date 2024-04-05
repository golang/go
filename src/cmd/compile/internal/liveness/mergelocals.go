// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package liveness

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/bitvec"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MergeLocalsState encapsulates information about which AUTO
// (stack-allocated) variables within a function can be safely
// merged/overlapped, e.g. share a stack slot with some other auto).
// An instance of MergeLocalsState is produced by MergeLocals() below
// and then consumed in ssagen.AllocFrame. The map 'partition' contains
// entries of the form <N,SL> where N is an *ir.Name and SL is a slice
// holding the indices (within 'vars') of other variables that share the
// same slot. For example, if a function contains five variables where
// v1/v2/v3 are safe to overlap and v4/v5 are safe to overlap, the
// MergeLocalsState content might look like
//
//	vars: [v1, v2, v3, v4, v5]
//	partition: v1 -> [1, 0, 2], v2 -> [1, 0, 2], v3 -> [1, 0, 2]
//	           v4 -> [3, 4], v5 -> [3, 4]
//
// A nil MergeLocalsState indicates that no local variables meet the
// necessary criteria for overlap.
type MergeLocalsState struct {
	// contains auto vars that participate in overlapping
	vars []*ir.Name
	// maps auto variable to overlap partition
	partition map[*ir.Name][]int
}

// candRegion is a sub-range (start, end) corresponding to an interval
// [st,en] within the list of candidate variables.
type candRegion struct {
	st, en int
}

// MergeLocals analyzes the specified ssa function f to determine which
// of its auto variables can safely share the same stack slot, returning
// a state object that describes how the overlap should be done.
func MergeLocals(fn *ir.Func, f *ssa.Func) *MergeLocalsState {
	cands, idx, regions := collectMergeCandidates(fn)
	if len(regions) == 0 {
		return nil
	}
	lv := newliveness(fn, f, cands, idx, 0)

	// If we have a local variable such as "r2" below that's written
	// but then not read, something like:
	//
	//      vardef r1
	//      r1.x = ...
	//      vardef r2
	//      r2.x = 0
	//      r2.y = ...
	//      <call foo>
	//      // no subsequent use of r2
	//      ... = r1.x
	//
	// then for the purpose of calculating stack maps at the call, we
	// can ignore "r2" completely during liveness analysis for stack
	// maps, however for stack slock merging we most definitely want
	// to treat the writes as "uses".
	lv.conservativeWrites = true

	lv.prologue()
	lv.solve()
	cs := &cstate{
		fn:        fn,
		ibuilders: make([]IntervalsBuilder, len(cands)),
	}
	computeIntervals(lv, cs)
	rv := performMerging(lv, cs, regions)
	if err := rv.check(); err != nil {
		base.FatalfAt(fn.Pos(), "invalid mergelocals state: %v", err)
	}
	return rv
}

// Subsumed returns whether variable n is subsumed, e.g. appears
// in an overlap position but is not the leader in that partition.
func (mls *MergeLocalsState) Subsumed(n *ir.Name) bool {
	if sl, ok := mls.partition[n]; ok && mls.vars[sl[0]] != n {
		return true
	}
	return false
}

// IsLeader returns whether a variable n is the leader (first element)
// in a sharing partition.
func (mls *MergeLocalsState) IsLeader(n *ir.Name) bool {
	if sl, ok := mls.partition[n]; ok && mls.vars[sl[0]] == n {
		return true
	}
	return false
}

// Leader returns the leader variable for subsumed var n.
func (mls *MergeLocalsState) Leader(n *ir.Name) *ir.Name {
	if sl, ok := mls.partition[n]; ok {
		if mls.vars[sl[0]] == n {
			panic("variable is not subsumed")
		}
		return mls.vars[sl[0]]
	}
	panic("not a merge candidate")
}

// Followers writes a list of the followers for leader n into the slice tmp.
func (mls *MergeLocalsState) Followers(n *ir.Name, tmp []*ir.Name) []*ir.Name {
	tmp = tmp[:0]
	sl, ok := mls.partition[n]
	if !ok {
		panic("no entry for leader")
	}
	if mls.vars[sl[0]] != n {
		panic("followers invoked on subsumed var")
	}
	for _, k := range sl[1:] {
		tmp = append(tmp, mls.vars[k])
	}
	sort.SliceStable(tmp, func(i, j int) bool {
		return tmp[i].Sym().Name < tmp[j].Sym().Name
	})
	return tmp
}

// EstSavings returns the estimated reduction in stack size (number of bytes) for
// the given merge locals state via a pair of ints, the first for non-pointer types and the second for pointer types.
func (mls *MergeLocalsState) EstSavings() (int, int) {
	totnp := 0
	totp := 0
	for n := range mls.partition {
		if mls.Subsumed(n) {
			sz := int(n.Type().Size())
			if n.Type().HasPointers() {
				totp += sz
			} else {
				totnp += sz
			}
		}
	}
	return totnp, totp
}

// check tests for various inconsistencies and problems in mls,
// returning an error if any problems are found.
func (mls *MergeLocalsState) check() error {
	if mls == nil {
		return nil
	}
	used := make(map[int]bool)
	seenv := make(map[*ir.Name]int)
	for ii, v := range mls.vars {
		if prev, ok := seenv[v]; ok {
			return fmt.Errorf("duplicate var %q in vslots: %d and %d\n",
				v.Sym().Name, ii, prev)
		}
		seenv[v] = ii
	}
	for k, sl := range mls.partition {
		// length of slice value needs to be more than 1
		if len(sl) < 2 {
			return fmt.Errorf("k=%q v=%+v slice len %d invalid",
				k.Sym().Name, sl, len(sl))
		}
		// values in the slice need to be var indices
		for i, v := range sl {
			if v < 0 || v > len(mls.vars)-1 {
				return fmt.Errorf("k=%q v=+%v slpos %d vslot %d out of range of m.v", k.Sym().Name, sl, i, v)
			}
		}
	}
	for k, sl := range mls.partition {
		foundk := false
		for i, v := range sl {
			vv := mls.vars[v]
			if i == 0 {
				if !mls.IsLeader(vv) {
					return fmt.Errorf("k=%s v=+%v slpos 0 vslot %d IsLeader(%q) is false should be true", k.Sym().Name, sl, v, vv.Sym().Name)
				}
			} else {
				if !mls.Subsumed(vv) {
					return fmt.Errorf("k=%s v=+%v slpos %d vslot %d Subsumed(%q) is false should be true", k.Sym().Name, sl, i, v, vv.Sym().Name)
				}
				if mls.Leader(vv) != mls.vars[sl[0]] {
					return fmt.Errorf("k=%s v=+%v slpos %d vslot %d Leader(%q) got %v want %v", k.Sym().Name, sl, i, v, vv.Sym().Name, mls.Leader(vv), mls.vars[sl[0]])
				}
			}
			if vv == k {
				foundk = true
				if used[v] {
					return fmt.Errorf("k=%s v=+%v val slice used violation at slpos %d vslot %d", k.Sym().Name, sl, i, v)
				}
				used[v] = true
			}
		}
		if !foundk {
			return fmt.Errorf("k=%s v=+%v slice value missing k", k.Sym().Name, sl)
		}
	}
	for i := range used {
		if !used[i] {
			return fmt.Errorf("pos %d var %q unused", i, mls.vars[i])
		}
	}
	return nil
}

func (mls *MergeLocalsState) String() string {
	var leaders []*ir.Name
	for n, sl := range mls.partition {
		if n == mls.vars[sl[0]] {
			leaders = append(leaders, n)
		}
	}
	sort.Slice(leaders, func(i, j int) bool {
		return leaders[i].Sym().Name < leaders[j].Sym().Name
	})
	var sb strings.Builder
	for _, n := range leaders {
		sb.WriteString(n.Sym().Name + ":")
		sl := mls.partition[n]
		for _, k := range sl[1:] {
			n := mls.vars[k]
			sb.WriteString(" " + n.Sym().Name)
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// collectMergeCandidates visits all of the AUTO vars declared in
// function fn and returns a list of candidate variables for merging /
// overlapping. Return values are: 1) a slice of ir.Name's
// corresponding to the candidates, 2) a map that maps ir.Name to slot
// in the slice, and 3) a slice containing regions (start/end pairs)
// corresponding to variables that could be overlapped provided that
// their lifetimes are disjoint.
func collectMergeCandidates(fn *ir.Func) ([]*ir.Name, map[*ir.Name]int32, []candRegion) {
	m := make(map[*ir.Name]int32)
	var cands []*ir.Name
	var regions []candRegion

	// Collect up the available set of appropriate AUTOs in the
	// function as a first step.
	for _, n := range fn.Dcl {
		if !n.Used() {
			continue
		}
		if !ssa.IsMergeCandidate(n) {
			continue
		}
		cands = append(cands, n)
	}
	if len(cands) < 2 {
		return nil, nil, nil
	}

	// Sort by pointerness, size, and then name.
	sort.SliceStable(cands, func(i, j int) bool {
		ci, cj := cands[i], cands[j]
		ihp, jhp := 0, 0
		var ilsym, jlsym *obj.LSym
		if ci.Type().HasPointers() {
			ihp = 1
			ilsym, _, _ = reflectdata.GCSym(ci.Type())
		}
		if cj.Type().HasPointers() {
			jhp = 1
			jlsym, _, _ = reflectdata.GCSym(cj.Type())
		}
		if ihp != jhp {
			return ihp < jhp
		}
		if ci.Type().Size() != cj.Type().Size() {
			return ci.Type().Size() < cj.Type().Size()
		}
		if ihp != 0 && jhp != 0 && ilsym != jlsym {
			// FIXME: find less clunky way to do this
			return fmt.Sprintf("%v", ilsym) < fmt.Sprintf("%v", jlsym)
		}
		if ci.Sym().Name != cj.Sym().Name {
			return ci.Sym().Name < cj.Sym().Name
		}
		return fmt.Sprintf("%v", ci.Pos()) < fmt.Sprintf("%v", ci.Pos())
	})

	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "=-= raw cand list for func %v:\n", fn)
		for i := range cands {
			dumpCand(cands[i], i)
		}
	}

	// Now generate a pruned candidate list-- we only want to return a
	// non-empty list if there is some possibility of overlapping two
	// vars.
	var pruned []*ir.Name
	st := 0
	for {
		en := nextRegion(cands, st)
		if en == -1 {
			break
		}
		if st == en {
			// region has just one element, we can skip it
			st++
			continue
		}
		pst := len(pruned)
		pen := pst + (en - st)
		if base.Debug.MergeLocalsTrace > 1 {
			fmt.Fprintf(os.Stderr, "=-= add part %d -> %d\n", pst, pen)
		}

		// non-empty region, add to pruned
		pruned = append(pruned, cands[st:en+1]...)
		regions = append(regions, candRegion{st: pst, en: pen})
		st = en + 1
	}
	if len(pruned) < 2 {
		return nil, nil, nil
	}
	for i, n := range pruned {
		m[n] = int32(i)
	}

	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "=-= pruned candidate list for func %v:\n", fn)
		for i := range pruned {
			dumpCand(pruned[i], i)
		}
	}
	return pruned, m, regions
}

// nextRegion starts at location idx and walks forward in the cands
// slice looking for variables that are "compatible" (overlappable)
// with the variable at position idx; it returns the end of the new
// region (range of compatible variables starting at idx).
func nextRegion(cands []*ir.Name, idx int) int {
	n := len(cands)
	if idx >= n {
		return -1
	}
	c0 := cands[idx]
	hp0 := c0.Type().HasPointers()
	for j := idx + 1; j < n; j++ {
		cj := cands[j]
		hpj := cj.Type().HasPointers()
		ok := true
		if hp0 {
			if !hpj || c0.Type().Size() != cj.Type().Size() {
				return j - 1
			}
			// GC shape must match if both types have pointers.
			gcsym0, _, _ := reflectdata.GCSym(c0.Type())
			gcsymj, _, _ := reflectdata.GCSym(cj.Type())
			if gcsym0 != gcsymj {
				return j - 1
			}
		} else {
			// If no pointers, match size only.
			if !ok || hp0 != hpj || c0.Type().Size() != cj.Type().Size() {
				return j - 1
			}
		}
	}
	return n - 1
}

type cstate struct {
	fn        *ir.Func
	ibuilders []IntervalsBuilder
}

// mergeVisitRegion tries to perform overlapping of variables with a
// given subrange of cands described by st and en (indices into our
// candidate var list), where the variables within this range have
// already been determined to be compatible with respect to type,
// size, etc. Overlapping is done in a a greedy fashion: we select the
// first element in the st->en range, then walk the rest of the
// elements adding in vars whose lifetimes don't overlap with the
// first element, then repeat the process until we run out of work to do.
func (mls *MergeLocalsState) mergeVisitRegion(lv *liveness, ivs []Intervals, st, en int) {
	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "=-= mergeVisitRegion(st=%d, en=%d)\n", st, en)
	}
	n := en - st + 1
	used := bitvec.New(int32(n))

	nxt := func(slot int) int {
		for c := slot - st; c < n; c++ {
			if used.Get(int32(c)) {
				continue
			}
			return c + st
		}
		return -1
	}

	navail := n
	cands := lv.vars
	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "  =-= navail = %d\n", navail)
	}
	for navail >= 2 {
		leader := nxt(st)
		used.Set(int32(leader - st))
		navail--

		if base.Debug.MergeLocalsTrace > 1 {
			fmt.Fprintf(os.Stderr, "  =-= begin leader %d used=%s\n", leader,
				used.String())
		}
		elems := []int{leader}
		lints := ivs[leader]

		for succ := nxt(leader + 1); succ != -1; succ = nxt(succ + 1) {

			// Skip if de-selected by merge locals hash.
			if base.Debug.MergeLocalsHash != "" {
				if !base.MergeLocalsHash.MatchPosWithInfo(cands[succ].Pos(), "mergelocals", nil) {
					continue
				}
			}
			// Skip if already used.
			if used.Get(int32(succ - st)) {
				continue
			}
			if base.Debug.MergeLocalsTrace > 1 {
				fmt.Fprintf(os.Stderr, "  =-= overlap of %d[%v] {%s} with %d[%v] {%s} is: %v\n", leader, cands[leader], lints.String(), succ, cands[succ], ivs[succ].String(), lints.Overlaps(ivs[succ]))
			}

			// Can we overlap leader with this var?
			if lints.Overlaps(ivs[succ]) {
				continue
			} else {
				// Add to overlap set.
				elems = append(elems, succ)
				lints = lints.Merge(ivs[succ])
			}
		}
		if len(elems) > 1 {
			// We found some things to overlap with leader. Add the
			// candidate elements to "vars" and update "partition".
			off := len(mls.vars)
			sl := make([]int, len(elems))
			for i, candslot := range elems {
				sl[i] = off + i
				mls.vars = append(mls.vars, cands[candslot])
				mls.partition[cands[candslot]] = sl
			}
			navail -= (len(elems) - 1)
			for i := range elems {
				used.Set(int32(elems[i] - st))
			}
			if base.Debug.MergeLocalsTrace > 1 {
				fmt.Fprintf(os.Stderr, "=-= overlapping %+v:\n", sl)
				for i := range sl {
					dumpCand(mls.vars[sl[i]], sl[i])
				}
				for i, v := range elems {
					fmt.Fprintf(os.Stderr, "=-= %d: sl=%d %s\n", i, v, ivs[v])
				}
			}
		}
	}
}

// performMerging carries out variable merging within each of the
// candidate ranges in regions, returning a state object
// that describes the variable overlaps.
func performMerging(lv *liveness, cs *cstate, regions []candRegion) *MergeLocalsState {
	cands := lv.vars
	mls := &MergeLocalsState{
		partition: make(map[*ir.Name][]int),
	}

	// Finish intervals construction.
	ivs := make([]Intervals, len(cands))
	for i := range cands {
		var err error
		ivs[i], err = cs.ibuilders[i].Finish()
		if err != nil {
			ninstr := 0
			if base.Debug.MergeLocalsTrace != 0 {
				iidx := 0
				for k := 0; k < len(lv.f.Blocks); k++ {
					b := lv.f.Blocks[k]
					fmt.Fprintf(os.Stderr, "\n")
					for _, v := range b.Values {
						fmt.Fprintf(os.Stderr, " b%d %d: %s\n", k, iidx, v.LongString())
						iidx++
						ninstr++
					}
				}
			}
			base.FatalfAt(cands[i].Pos(), "interval construct error for var %q in func %q (%d instrs): %v", cands[i].Sym().Name, ir.FuncName(cs.fn), ninstr, err)
			return nil
		}
	}

	// Dump state before attempting overlap.
	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "=-= cands live before overlap:\n")
		for i := range cands {
			c := cands[i]
			fmt.Fprintf(os.Stderr, "%d: %v sz=%d ivs=%s\n",
				i, c.Sym().Name, c.Type().Size(), ivs[i].String())
		}
		fmt.Fprintf(os.Stderr, "=-= regions (%d): ", len(regions))
		for _, cr := range regions {
			fmt.Fprintf(os.Stderr, " [%d,%d]", cr.st, cr.en)
		}
		fmt.Fprintf(os.Stderr, "\n")
	}

	if base.Debug.MergeLocalsTrace > 1 {
		fmt.Fprintf(os.Stderr, "=-= len(regions) = %d\n", len(regions))
	}

	// Apply a greedy merge/overlap strategy within each region
	// of compatible variables.
	for _, cr := range regions {
		mls.mergeVisitRegion(lv, ivs, cr.st, cr.en)
	}
	if len(mls.vars) == 0 {
		return nil
	}
	return mls
}

// computeIntervals performs a backwards sweep over the instructions
// of the function we're compiling, building up an Intervals object
// for each candidate variable by looking for upwards exposed uses
// and kills.
func computeIntervals(lv *liveness, cs *cstate) {
	nvars := int32(len(lv.vars))
	liveout := bitvec.New(nvars)

	if base.Debug.MergeLocalsDumpFunc != "" &&
		strings.HasSuffix(fmt.Sprintf("%v", cs.fn), base.Debug.MergeLocalsDumpFunc) {
		fmt.Fprintf(os.Stderr, "=-= mergelocalsdumpfunc %v:\n", cs.fn)
		ii := 0
		for k, b := range lv.f.Blocks {
			fmt.Fprintf(os.Stderr, "b%d:\n", k)
			for _, v := range b.Values {
				pos := base.Ctxt.PosTable.Pos(v.Pos)
				fmt.Fprintf(os.Stderr, "=-= %d L%d|C%d %s\n", ii, pos.RelLine(), pos.RelCol(), v.LongString())
				ii++
			}
		}
	}

	// Count instructions.
	ninstr := 0
	for _, b := range lv.f.Blocks {
		ninstr += len(b.Values)
	}
	// current instruction index during backwards walk
	iidx := ninstr - 1

	// Make a backwards pass over all blocks
	for k := len(lv.f.Blocks) - 1; k >= 0; k-- {
		b := lv.f.Blocks[k]
		be := lv.blockEffects(b)

		if base.Debug.MergeLocalsTrace > 2 {
			fmt.Fprintf(os.Stderr, "=-= liveout from tail of b%d: ", k)
			for j := range lv.vars {
				if be.liveout.Get(int32(j)) {
					fmt.Fprintf(os.Stderr, " %q", lv.vars[j].Sym().Name)
				}
			}
			fmt.Fprintf(os.Stderr, "\n")
		}

		// Take into account effects taking place at end of this basic
		// block by comparing our current live set with liveout for
		// the block. If a given var was not live before and is now
		// becoming live we need to mark this transition with a
		// builder "Live" call; similarly if a var was live before and
		// is now no longer live, we need a "Kill" call.
		for j := range lv.vars {
			isLive := liveout.Get(int32(j))
			blockLiveOut := be.liveout.Get(int32(j))
			if isLive {
				if !blockLiveOut {
					if base.Debug.MergeLocalsTrace > 2 {
						fmt.Fprintf(os.Stderr, "=+= at instr %d block boundary kill of %v\n", iidx, lv.vars[j])
					}
					cs.ibuilders[j].Kill(iidx)
				}
			} else if blockLiveOut {
				if base.Debug.MergeLocalsTrace > 2 {
					fmt.Fprintf(os.Stderr, "=+= at block-end instr %d %v becomes live\n",
						iidx, lv.vars[j])
				}
				cs.ibuilders[j].Live(iidx)
			}
		}

		// Set our working "currently live" set to the previously
		// computed live out set for the block.
		liveout.Copy(be.liveout)

		// Now walk backwards through this block.
		for i := len(b.Values) - 1; i >= 0; i-- {
			v := b.Values[i]

			if base.Debug.MergeLocalsTrace > 2 {
				fmt.Fprintf(os.Stderr, "=-= b%d instr %d: %s\n", k, iidx, v.LongString())
			}

			// Update liveness based on what we see happening in this
			// instruction.
			pos, e := lv.valueEffects(v)
			becomeslive := e&uevar != 0
			iskilled := e&varkill != 0
			if becomeslive && iskilled {
				// we do not ever expect to see both a kill and an
				// upwards exposed use given our size constraints.
				panic("should never happen")
			}
			if iskilled && liveout.Get(pos) {
				cs.ibuilders[pos].Kill(iidx)
				liveout.Unset(pos)
				if base.Debug.MergeLocalsTrace > 2 {
					fmt.Fprintf(os.Stderr, "=+= at instr %d kill of %v\n",
						iidx, lv.vars[pos])
				}
			} else if becomeslive && !liveout.Get(pos) {
				cs.ibuilders[pos].Live(iidx)
				liveout.Set(pos)
				if base.Debug.MergeLocalsTrace > 2 {
					fmt.Fprintf(os.Stderr, "=+= at instr %d upwards-exposed use of %v\n",
						iidx, lv.vars[pos])
				}
			}
			iidx--
		}

		if b == lv.f.Entry {
			for j, v := range lv.vars {
				if liveout.Get(int32(j)) {
					lv.f.Fatalf("%v %L recorded as live on entry",
						lv.fn.Nname, v)
				}
			}
		}
	}
	if iidx != -1 {
		panic("iidx underflow")
	}
}

func dumpCand(c *ir.Name, i int) {
	fmtFullPos := func(p src.XPos) string {
		var sb strings.Builder
		sep := ""
		base.Ctxt.AllPos(p, func(pos src.Pos) {
			fmt.Fprintf(&sb, sep)
			sep = "|"
			file := filepath.Base(pos.Filename())
			fmt.Fprintf(&sb, "%s:%d:%d", file, pos.Line(), pos.Col())
		})
		return sb.String()
	}
	fmt.Fprintf(os.Stderr, " %d: %s %q sz=%d hp=%v t=%v\n",
		i, fmtFullPos(c.Pos()), c.Sym().Name, c.Type().Size(),
		c.Type().HasPointers(), c.Type())
}

// for unit testing only.
func MakeMergeLocalsState(partition map[*ir.Name][]int, vars []*ir.Name) (*MergeLocalsState, error) {
	mls := &MergeLocalsState{partition: partition, vars: vars}
	if err := mls.check(); err != nil {
		return nil, err
	}
	return mls, nil
}
