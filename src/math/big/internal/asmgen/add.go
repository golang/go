// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

// addOrSubVV generates addVV or subVV,
// which do z, c = x ± y.
// The caller guarantees that len(z) == len(x) == len(y).
func addOrSubVV(a *Asm, name string) {
	f := a.Func("func " + name + "(z, x, y []Word) (c Word)")

	add := a.Add
	which := AddCarry
	if name == "subVV" {
		add = a.Sub
		which = SubCarry
	}

	n := f.Arg("z_len")
	p := f.Pipe()
	p.SetHint("y", HintMemOK) // allow y to be used from memory on x86
	p.Start(n, 1, 4)
	var c Reg
	if !a.Arch.CarrySafeLoop {
		// Carry smashed by loop tests; allocate and save in register
		// around unrolled blocks.
		c = a.Reg()
		a.Mov(a.Imm(0), c)
		a.EOL("clear saved carry")
		p.AtUnrollStart(func() { a.RestoreCarry(c); a.Free(c) })
		p.AtUnrollEnd(func() { a.Unfree(c); a.SaveCarry(c) })
	} else {
		// Carry preserved by loop; clear now, ahead of loop
		// (but after Start, which may have modified it).
		a.ClearCarry(which)
	}
	p.Loop(func(in, out [][]Reg) {
		for i, x := range in[0] {
			y := in[1][i]
			add(y, x, x, SetCarry|UseCarry)
		}
		p.StoreN(in[:1])
	})
	p.Done()

	// Copy carry to output.
	if c.Valid() {
		a.ConvertCarry(which, c)
	} else {
		c = a.RegHint(HintCarry)
		a.SaveConvertCarry(which, c)
	}
	f.StoreArg(c, "c")
	a.Free(c)
	a.Ret()
}
