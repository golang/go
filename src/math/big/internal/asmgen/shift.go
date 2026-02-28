// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

// shiftVU generates lshVU and rshVU, which do
// z, c = x << s and z, c = x >> s, for 0 < s < _W.
func shiftVU(a *Asm, name string) {
	// Because these routines can be called for z.Lsh(z, N) and z.Rsh(z, N),
	// the input and output slices may be aliased at different offsets.
	// For example (on 64-bit systems), during z.Lsh(z, 65), &z[0] == &x[1],
	// and during z.Rsh(z, 65), &z[1] == &x[0].
	// For left shift, we must process the slices from len(z)-1 down to 0,
	// so that we don't overwrite a word before we need to read it.
	// For right shift, we must process the slices from 0 up to len(z)-1.
	// The different traversals at least make the two cases more consistent,
	// since we're always delaying the output by one word compared
	// to the input.

	f := a.Func("func " + name + "(z, x []Word, s uint) (c Word)")

	// Check for no input early, since we need to start by reading 1 word.
	n := f.Arg("z_len")
	a.JmpZero(n, "ret0")

	// Start loop by reading first input word.
	s := f.ArgHint("s", HintShiftCount)
	p := f.Pipe()
	if name == "lshVU" {
		p.SetBackward()
	}
	unroll := []int{1, 4}
	if a.Arch == Arch386 {
		unroll = []int{1} // too few registers for more
		p.SetUseIndexCounter()
	}
	p.LoadPtrs(n)
	a.Comment("shift first word into carry")
	prev := p.LoadN(1)[0][0]

	// Decide how to shift. On systems with a wide shift (x86), use that.
	// Otherwise, we need shift by s and negative (reverse) shift by 64-s or 32-s.
	shift := a.Lsh
	shiftWide := a.LshWide
	negShift := a.Rsh
	negShiftReg := a.RshReg
	if name == "rshVU" {
		shift = a.Rsh
		shiftWide = a.RshWide
		negShift = a.Lsh
		negShiftReg = a.LshReg
	}
	if a.Arch.HasShiftWide() {
		// Use wide shift to avoid needing negative shifts.
		// The invariant is that prev holds the previous word (not shifted at all),
		// to be used as input into the wide shift.
		// After the loop finishes, prev holds the final output word to be written.
		c := a.Reg()
		shiftWide(s, prev, a.Imm(0), c)
		f.StoreArg(c, "c")
		a.Free(c)
		a.Comment("shift remaining words")
		p.Start(n, unroll...)
		p.Loop(func(in [][]Reg, out [][]Reg) {
			// We reuse the input registers as output, delayed one cycle; prev is the first output.
			// After writing the outputs to memory, we can copy the final x value into prev
			// for the next iteration.
			old := prev
			for i, x := range in[0] {
				shiftWide(s, x, old, old)
				out[0][i] = old
				old = x
			}
			p.StoreN(out)
			a.Mov(old, prev)
		})
		a.Comment("store final shifted bits")
		shift(s, prev, prev)
	} else {
		// Construct values from x << s and x >> (64-s).
		// After the first word has been processed, the invariant is that
		// prev holds x << s, to be used as the high bits of the next output word,
		// once we find the low bits after reading the next input word.
		// After the loop finishes, prev holds the final output word to be written.
		sNeg := a.Reg()
		a.Mov(a.Imm(a.Arch.WordBits), sNeg)
		a.Sub(s, sNeg, sNeg, SmashCarry)
		c := a.Reg()
		negShift(sNeg, prev, c)
		shift(s, prev, prev)
		f.StoreArg(c, "c")
		a.Free(c)
		a.Comment("shift remaining words")
		p.Start(n, unroll...)
		p.Loop(func(in, out [][]Reg) {
			if a.HasRegShift() {
				// ARM (32-bit) allows shifts in most arithmetic expressions,
				// including OR, letting us combine the negShift and a.Or.
				// The simplest way to manage the registers is to do StoreN for
				// one output at a time, and since we don't use multi-register
				// stores on ARM, that doesn't hurt us.
				out[0] = out[0][:1]
				for _, x := range in[0] {
					a.Or(negShiftReg(sNeg, x), prev, prev)
					out[0][0] = prev
					p.StoreN(out)
					shift(s, x, prev)
				}
				return
			}
			// We reuse the input registers as output, delayed one cycle; z0 is the first output.
			z0 := a.Reg()
			z := z0
			for i, x := range in[0] {
				negShift(sNeg, x, z)
				a.Or(prev, z, z)
				shift(s, x, prev)
				out[0][i] = z
				z = x
			}
			p.StoreN(out)
		})
		a.Comment("store final shifted bits")
	}
	p.StoreN([][]Reg{{prev}})
	p.Done()
	a.Free(s)
	a.Ret()

	// Return 0, used from above.
	a.Label("ret0")
	f.StoreArg(a.Imm(0), "c")
	a.Ret()
}
