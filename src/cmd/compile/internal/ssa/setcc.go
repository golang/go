// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/ssa/ssabase"

// setccZeroExtend rewrites the amd64 boolean materialization idiom
//
//	CMPQ    x, y
//	SETcc   AX      // writes AL, leaving AX's upper bits alone
//	MOVBLZX AX, AX  // ... so they have to be cleared afterwards
//
// into the form the Intel optimization manual recommends, zeroing the
// register ahead of the comparison instead:
//
//	XORL    AX, AX
//	CMPQ    x, y
//	SETcc   AX
//
// Both forms leave 0 or 1 in AX, but the second is a byte shorter, spends
// one fewer execution slot (XORL is a zeroing idiom the renamer resolves,
// where MOVBLZX is a real uop), and drops the false dependency that the
// SETcc's partial-register write otherwise carries on AX's previous value.
//
// This runs after regalloc because the rewrite is a statement about a
// machine register, not about a value: it is legal exactly when the
// register the SETcc writes is dead across the comparison it is being
// hoisted over. Before register assignment that is not yet knowable, and
// forcing it as a constraint would cost a register instead of saving an
// instruction.
func setccZeroExtend(f *Func) {
	if f.Config.arch != "amd64" {
		return
	}
	for _, b := range f.Blocks {
		for i := len(b.Values) - 1; i > 0; i-- {
			ext := b.Values[i]
			if ext.Op != OpAMD64MOVBQZX {
				continue
			}
			// The SETcc must feed the widening directly. Anything in
			// between would have to be checked for a flags clobber, and
			// this is the shape the idiom actually takes.
			set := b.Values[i-1]
			if set != ext.Args[0] || !isSETcc(set.Op) {
				continue
			}
			// The widening must be in place: if the two land in different
			// registers this is a move, and pre-zeroing proves nothing
			// about the register the SETcc writes.
			reg, ok := f.getHome(ext.ID).(*ssabase.Register)
			if !ok || f.getHome(set.ID) != Location(reg) {
				continue
			}

			// Find the instruction that sets the flags the SETcc reads.
			// Tuple selectors emit no code of their own, so for those the
			// flag setter is the tuple-producing instruction.
			flags := set.Args[0]
			for flags.Op == OpSelect0 || flags.Op == OpSelect1 {
				flags = flags.Args[0]
			}
			if flags.Block != b {
				// Flags generated in a predecessor: a path that skips the
				// zeroing could still reach the SETcc.
				continue
			}
			j := -1
			for k := i - 2; k >= 0; k-- {
				if b.Values[k] == flags {
					j = k
					break
				}
			}
			if j < 0 {
				continue
			}

			// The zeroing lands in front of the flag setter, so the
			// register has to be free from there down to the SETcc that
			// redefines it. Nothing may read the value it displaces, and
			// nothing may write the zero back to junk.
			free := true
			for k := j; k < i-1; k++ {
				if valueUsesReg(f, b.Values[k], reg) {
					free = false
					break
				}
			}
			if !free {
				continue
			}

			// The zeroing clobbers flags. Where the flag setter consumes
			// flags itself -- the ADCQ/SBBQ links of a carry chain -- they
			// are still live in front of it, and the backend would have to
			// fall back on a 5-byte MOVL $0 that saves neither space nor a
			// uop over the MOVBLZX it replaces. Leave those alone.
			if flagsLiveBefore(b, j) {
				continue
			}

			// MOVLconst 0 is emitted as XORL reg, reg.
			zero := b.NewValue0I(flags.Pos, OpAMD64MOVLconst, f.Config.Types.UInt32, 0)
			f.setHome(zero, reg)
			copy(b.Values[j+1:], b.Values[j:len(b.Values)-1])
			b.Values[j] = zero

			// The SETcc now merges into a zeroed register, so the widening
			// is redundant. A copy within one register emits nothing.
			ext.Op = OpCopy

			// The insertion shifted everything from j up by one, so the
			// next value to examine has moved from i-2 to i-1: the loop's
			// own decrement lands on it. Two SETcc reading one comparison
			// each get their own zeroing this way.
		}
	}
}

// isSETcc reports whether op is an amd64 SETcc that writes its condition to
// the low byte of a register with a single instruction. The float-compare
// pair SETEQF/SETNEF is excluded: they expand to three instructions ending
// in an ANDL of two junk-topped registers, which zeroing one input does not
// clean up.
func isSETcc(op Op) bool {
	switch op {
	case OpAMD64SETEQ, OpAMD64SETNE,
		OpAMD64SETL, OpAMD64SETLE,
		OpAMD64SETG, OpAMD64SETGE,
		OpAMD64SETGF, OpAMD64SETGEF,
		OpAMD64SETB, OpAMD64SETBE,
		OpAMD64SETORD, OpAMD64SETNAN,
		OpAMD64SETA, OpAMD64SETAE,
		OpAMD64SETO:
		return true
	}
	return false
}

// flagsLiveBefore reports whether a flags value is live entering
// b.Values[i]. It mirrors the backend's own scan (ssaMarkMoves), which is
// what decides whether a zero constant may be materialized with XORL, so
// the two always agree about a given insertion point.
func flagsLiveBefore(b *Block, i int) bool {
	live := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		live = live || c.Type.IsFlags()
	}
	for k := len(b.Values) - 1; k >= i; k-- {
		v := b.Values[k]
		if v.Type.IsFlags() {
			live = false
		}
		for _, a := range v.Args {
			if a.Type.IsFlags() {
				live = true
			}
		}
	}
	return live
}

// valueUsesReg reports whether v reads, writes, or clobbers register r under
// the assignment regalloc has already made.
func valueUsesReg(f *Func, v *Value, r *ssabase.Register) bool {
	if locationUsesReg(f.getHome(v.ID), r) {
		return true
	}
	for _, a := range v.Args {
		if locationUsesReg(f.getHome(a.ID), r) {
			return true
		}
	}
	if opcodeTable[v.Op].reg.clobbers.hasReg(register(r.Num)) {
		return true
	}
	return f.tempRegs[v.ID] == r
}

// locationUsesReg reports whether loc holds register r, including the
// multi-register homes of tuple- and results-typed values.
func locationUsesReg(loc Location, r *ssabase.Register) bool {
	switch loc := loc.(type) {
	case *ssabase.Register:
		return loc == r
	case LocPair:
		return loc[0] == Location(r) || loc[1] == Location(r)
	case LocResults:
		for _, l := range loc {
			if l == Location(r) {
				return true
			}
		}
	}
	return false
}
