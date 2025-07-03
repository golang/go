// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

// mulAddVWW generates mulAddVWW, which does z, c = x*m + a.
func mulAddVWW(a *Asm) {
	f := a.Func("func mulAddVWW(z, x []Word, m, a Word) (c Word)")

	if a.AltCarry().Valid() {
		addMulVirtualCarry(f, 0)
		return
	}
	addMul(f, "", "x", 0)
}

// addMulVVWW generates addMulVVWW which does z, c = x + y*m + a.
// (A more pedantic name would be addMulAddVVWW.)
func addMulVVWW(a *Asm) {
	f := a.Func("func addMulVVWW(z, x, y []Word, m, a Word) (c Word)")

	// If the architecture has virtual carries, emit that version unconditionally.
	if a.AltCarry().Valid() {
		addMulVirtualCarry(f, 1)
		return
	}

	// If the architecture optionally has two carries, test and emit both versions.
	if a.JmpEnable(OptionAltCarry, "altcarry") {
		regs := a.RegsUsed()
		addMul(f, "x", "y", 1)
		a.Label("altcarry")
		a.SetOption(OptionAltCarry, true)
		a.SetRegsUsed(regs)
		addMulAlt(f)
		a.SetOption(OptionAltCarry, false)
		return
	}

	// Otherwise emit the one-carry form.
	addMul(f, "x", "y", 1)
}

// Computing z = addsrc + m*mulsrc + a, we need:
//
//	for i := range z {
//		lo, hi := m * mulsrc[i]
//		lo, carry = bits.Add(lo, a, 0)
//		lo, carryAlt = bits.Add(lo, addsrc[i], 0)
//		z[i] = lo
//		a = hi + carry + carryAlt  // cannot overflow
//	}
//
// The final addition cannot overflow because after processing N words,
// the maximum possible value is (for a 64-bit system):
//
//	  (2**64N - 1) + (2**64 - 1)*(2**64N - 1) + (2**64 - 1)
//	= (2**64)*(2**64N - 1) + (2**64 - 1)
//	= 2**64(N+1) - 1,
//
// which fits in N+1 words (the high order one being the new value of a).
//
// (For example, with 3 decimal words, 999 + 9*999 + 9 = 999*10 + 9 = 9999.)
//
// If we unroll the loop a bit, then we can chain the carries in two passes.
// Consider:
//
//	lo0, hi0 := m * mulsrc[i]
//	lo0, carry = bits.Add(lo0, a, 0)
//	lo0, carryAlt = bits.Add(lo0, addsrc[i], 0)
//	z[i] = lo0
//	a = hi + carry + carryAlt // cannot overflow
//
//	lo1, hi1 := m * mulsrc[i]
//	lo1, carry = bits.Add(lo1, a, 0)
//	lo1, carryAlt = bits.Add(lo1, addsrc[i], 0)
//	z[i] = lo1
//	a = hi + carry + carryAlt // cannot overflow
//
//	lo2, hi2 := m * mulsrc[i]
//	lo2, carry = bits.Add(lo2, a, 0)
//	lo2, carryAlt = bits.Add(lo2, addsrc[i], 0)
//	z[i] = lo2
//	a = hi + carry + carryAlt // cannot overflow
//
//	lo3, hi3 := m * mulsrc[i]
//	lo3, carry = bits.Add(lo3, a, 0)
//	lo3, carryAlt = bits.Add(lo3, addsrc[i], 0)
//	z[i] = lo3
//	a = hi + carry + carryAlt // cannot overflow
//
// There are three ways we can optimize this sequence.
//
// (1) Reordering, we can chain carries so that we can use one hardware carry flag
// but amortize the cost of saving and restoring it across multiple instructions:
//
//	// multiply
//	lo0, hi0 := m * mulsrc[i]
//	lo1, hi1 := m * mulsrc[i+1]
//	lo2, hi2 := m * mulsrc[i+2]
//	lo3, hi3 := m * mulsrc[i+3]
//
//	lo0, carry = bits.Add(lo0, a, 0)
//	lo1, carry = bits.Add(lo1, hi0, carry)
//	lo2, carry = bits.Add(lo2, hi1, carry)
//	lo3, carry = bits.Add(lo3, hi2, carry)
//	a = hi3 + carry // cannot overflow
//
//	// add
//	lo0, carryAlt = bits.Add(lo0, addsrc[i], 0)
//	lo1, carryAlt = bits.Add(lo1, addsrc[i+1], carryAlt)
//	lo2, carryAlt = bits.Add(lo2, addsrc[i+2], carryAlt)
//	lo3, carryAlt = bits.Add(lo3, addrsc[i+3], carryAlt)
//	a = a + carryAlt // cannot overflow
//
//	z[i] = lo0
//	z[i+1] = lo1
//	z[i+2] = lo2
//	z[i+3] = lo3
//
// addMul takes this approach, using the hardware carry flag
// first for carry and then for carryAlt.
//
// (2) addMulAlt assumes there are two hardware carry flags available.
// It dedicates one each to carry and carryAlt, so that a multi-block
// unrolling can keep the flags in hardware across all the blocks.
// So even if the block size is 1, the code can do:
//
//	// multiply and add
//	lo0, hi0 := m * mulsrc[i]
//	lo0, carry = bits.Add(lo0, a, 0)
//	lo0, carryAlt = bits.Add(lo0, addsrc[i], 0)
//	z[i] = lo0
//
//	lo1, hi1 := m * mulsrc[i+1]
//	lo1, carry = bits.Add(lo1, hi0, carry)
//	lo1, carryAlt = bits.Add(lo1, addsrc[i+1], carryAlt)
//	z[i+1] = lo1
//
//	lo2, hi2 := m * mulsrc[i+2]
//	lo2, carry = bits.Add(lo2, hi1, carry)
//	lo2, carryAlt = bits.Add(lo2, addsrc[i+2], carryAlt)
//	z[i+2] = lo2
//
//	lo3, hi3 := m * mulsrc[i+3]
//	lo3, carry = bits.Add(lo3, hi2, carry)
//	lo3, carryAlt = bits.Add(lo3, addrsc[i+3], carryAlt)
//	z[i+3] = lo2
//
//	a = hi3 + carry + carryAlt // cannot overflow
//
// (3) addMulVirtualCarry optimizes for systems with explicitly computed carry bits
// (loong64, mips, riscv64), cutting the number of actual instructions almost by half.
// Look again at the original word-at-a-time version:
//
//	lo1, hi1 := m * mulsrc[i]
//	lo1, carry = bits.Add(lo1, a, 0)
//	lo1, carryAlt = bits.Add(lo1, addsrc[i], 0)
//	z[i] = lo1
//	a = hi + carry + carryAlt // cannot overflow
//
// Although it uses four adds per word, those are cheap adds: the two bits.Add adds
// use two instructions each (ADD+SLTU) and the final + adds only use one ADD each,
// for a total of 6 instructions per word. In contrast, the middle stanzas in (2) use
// only two “adds” per word, but these are SetCarry|UseCarry adds, which compile to
// five instruction each, for a total of 10 instructions per word. So the word-at-a-time
// loop is actually better. And we can reorder things slightly to use only a single carry bit:
//
//	lo1, hi1 := m * mulsrc[i]
//	lo1, carry = bits.Add(lo1, a, 0)
//	a = hi + carry
//	lo1, carry = bits.Add(lo1, addsrc[i], 0)
//	a = a + carry
//	z[i] = lo1
func addMul(f *Func, addsrc, mulsrc string, mulIndex int) {
	a := f.Asm
	mh := HintNone
	if a.Arch == Arch386 && addsrc != "" {
		mh = HintMemOK // too few registers otherwise
	}
	m := f.ArgHint("m", mh)
	c := f.Arg("a")
	n := f.Arg("z_len")

	p := f.Pipe()
	if addsrc != "" {
		p.SetHint(addsrc, HintMemOK)
	}
	p.SetHint(mulsrc, HintMulSrc)
	unroll := []int{1, 4}
	switch a.Arch {
	case Arch386:
		unroll = []int{1} // too few registers
	case ArchARM:
		p.SetMaxColumns(2) // too few registers (but more than 386)
	case ArchARM64:
		unroll = []int{1, 8} // 5% speedup on c4as16
	}

	// See the large comment above for an explanation of the code being generated.
	// This is optimization strategy 1.
	p.Start(n, unroll...)
	p.Loop(func(in, out [][]Reg) {
		a.Comment("multiply")
		prev := c
		flag := SetCarry
		for i, x := range in[mulIndex] {
			hi := a.RegHint(HintMulHi)
			a.MulWide(m, x, x, hi)
			a.Add(prev, x, x, flag)
			flag = UseCarry | SetCarry
			if prev != c {
				a.Free(prev)
			}
			out[0][i] = x
			prev = hi
		}
		a.Add(a.Imm(0), prev, c, UseCarry|SmashCarry)
		if addsrc != "" {
			a.Comment("add")
			flag := SetCarry
			for i, x := range in[0] {
				a.Add(x, out[0][i], out[0][i], flag)
				flag = UseCarry | SetCarry
			}
			a.Add(a.Imm(0), c, c, UseCarry|SmashCarry)
		}
		p.StoreN(out)
	})

	f.StoreArg(c, "c")
	a.Ret()
}

func addMulAlt(f *Func) {
	a := f.Asm
	m := f.ArgHint("m", HintMulSrc)
	c := f.Arg("a")
	n := f.Arg("z_len")

	// On amd64, we need a non-immediate for the AtUnrollEnd adds.
	r0 := a.ZR()
	if !r0.Valid() {
		r0 = a.Reg()
		a.Mov(a.Imm(0), r0)
	}

	p := f.Pipe()
	p.SetLabel("alt")
	p.SetHint("x", HintMemOK)
	p.SetHint("y", HintMemOK)
	if a.Arch == ArchAMD64 {
		p.SetMaxColumns(2)
	}

	// See the large comment above for an explanation of the code being generated.
	// This is optimization strategy (2).
	var hi Reg
	prev := c
	p.Start(n, 1, 8)
	p.AtUnrollStart(func() {
		a.Comment("multiply and add")
		a.ClearCarry(AddCarry | AltCarry)
		a.ClearCarry(AddCarry)
		hi = a.Reg()
	})
	p.AtUnrollEnd(func() {
		a.Add(r0, prev, c, UseCarry|SmashCarry)
		a.Add(r0, c, c, UseCarry|SmashCarry|AltCarry)
		prev = c
	})
	p.Loop(func(in, out [][]Reg) {
		for i, y := range in[1] {
			x := in[0][i]
			lo := y
			if lo.IsMem() {
				lo = a.Reg()
			}
			a.MulWide(m, y, lo, hi)
			a.Add(prev, lo, lo, UseCarry|SetCarry)
			a.Add(x, lo, lo, UseCarry|SetCarry|AltCarry)
			out[0][i] = lo
			prev, hi = hi, prev
		}
		p.StoreN(out)
	})

	f.StoreArg(c, "c")
	a.Ret()
}

func addMulVirtualCarry(f *Func, mulIndex int) {
	a := f.Asm
	m := f.Arg("m")
	c := f.Arg("a")
	n := f.Arg("z_len")

	// See the large comment above for an explanation of the code being generated.
	// This is optimization strategy (3).
	p := f.Pipe()
	p.Start(n, 1, 4)
	p.Loop(func(in, out [][]Reg) {
		a.Comment("synthetic carry, one column at a time")
		lo, hi := a.Reg(), a.Reg()
		for i, x := range in[mulIndex] {
			a.MulWide(m, x, lo, hi)
			if mulIndex == 1 {
				a.Add(in[0][i], lo, lo, SetCarry)
				a.Add(a.Imm(0), hi, hi, UseCarry|SmashCarry)
			}
			a.Add(c, lo, x, SetCarry)
			a.Add(a.Imm(0), hi, c, UseCarry|SmashCarry)
			out[0][i] = x
		}
		p.StoreN(out)
	})
	f.StoreArg(c, "c")
	a.Ret()
}
