// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

var ArchAMD64 = &Arch{
	Name:      "amd64",
	WordBits:  64,
	WordBytes: 8,

	regs: []string{
		"BX", "SI", "DI",
		"R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",
		"AX", "DX", "CX", // last to leave available for hinted allocation
	},
	op3:              x86Op3,
	hint:             x86Hint,
	memOK:            true,
	subCarryIsBorrow: true,

	// Note: Not setting memIndex, because code generally runs faster
	// if we avoid the use of scaled-index memory references,
	// particularly in ADX instructions.

	options: map[Option]func(*Asm, string){
		OptionAltCarry: amd64JmpADX,
	},

	mov:      "MOVQ",
	adds:     "ADDQ",
	adcs:     "ADCQ",
	subs:     "SUBQ",
	sbcs:     "SBBQ",
	lsh:      "SHLQ",
	lshd:     "SHLQ",
	rsh:      "SHRQ",
	rshd:     "SHRQ",
	and:      "ANDQ",
	or:       "ORQ",
	xor:      "XORQ",
	neg:      "NEGQ",
	lea:      "LEAQ",
	addF:     amd64Add,
	mulWideF: x86MulWide,

	addWords: "LEAQ (%[2]s)(%[1]s*8), %[3]s",

	jmpZero:       "TESTQ %[1]s, %[1]s; JZ %[2]s",
	jmpNonZero:    "TESTQ %[1]s, %[1]s; JNZ %[2]s",
	loopBottom:    "SUBQ $1, %[1]s; JNZ %[2]s",
	loopBottomNeg: "ADDQ $1, %[1]s; JNZ %[2]s",
}

func amd64JmpADX(a *Asm, label string) {
	a.Printf("\tCMPB ·hasADX(SB), $0; JNZ %s\n", label)
}

func amd64Add(a *Asm, src1, src2 Reg, dst Reg, carry Carry) bool {
	if a.Enabled(OptionAltCarry) {
		// If OptionAltCarry is enabled, the generator is emitting ADD instructions
		// both with and without the AltCarry flag set; the AltCarry flag means to
		// use ADOX. Otherwise we have to use ADCX.
		// Using regular ADD/ADC would smash both carry flags,
		// so we reject anything we can't handled with ADCX/ADOX.
		if carry&UseCarry != 0 && carry&(SetCarry|SmashCarry) != 0 {
			if carry&AltCarry != 0 {
				a.op3("ADOXQ", src1, src2, dst)
			} else {
				a.op3("ADCXQ", src1, src2, dst)
			}
			return true
		}
		if carry&(SetCarry|UseCarry) == SetCarry && a.IsZero(src1) && src2 == dst {
			// Clearing carry flag. Caller will add EOL comment.
			a.Printf("\tTESTQ AX, AX\n")
			return true
		}
		if carry != KeepCarry {
			a.Fatalf("unsupported carry")
		}
	}
	return false
}

// The x86-prefixed functions are shared with Arch386 in 386.go.

func x86Op3(name string) bool {
	// As far as a.op3 is concerned, there are no 3-op instructions.
	// (We print instructions like MULX ourselves.)
	return false
}

func x86Hint(a *Asm, h Hint) string {
	switch h {
	case HintShiftCount:
		return "CX"
	case HintMulSrc:
		if a.Enabled(OptionAltCarry) { // using MULX
			return "DX"
		}
		return "AX"
	case HintMulHi:
		if a.Enabled(OptionAltCarry) { // using MULX
			return ""
		}
		return "DX"
	}
	return ""
}

func x86Suffix(a *Asm) string {
	// Note: Not using a.Arch == Arch386 to avoid init cycle.
	if a.Arch.Name == "386" {
		return "L"
	}
	return "Q"
}

func x86MulWide(a *Asm, src1, src2, dstlo, dsthi Reg) {
	if a.Enabled(OptionAltCarry) {
		// Using ADCX/ADOX; use MULX to avoid clearing carry flag.
		if src1.name != "DX" {
			if src2.name != "DX" {
				a.Fatalf("mul src1 or src2 must be DX")
			}
			src2 = src1
		}
		a.Printf("\tMULXQ %s, %s, %s\n", src2, dstlo, dsthi)
		return
	}

	if src1.name != "AX" {
		if src2.name != "AX" {
			a.Fatalf("mulwide src1 or src2 must be AX")
		}
		src2 = src1
	}
	if dstlo.name != "AX" {
		a.Fatalf("mulwide dstlo must be AX")
	}
	if dsthi.name != "DX" {
		a.Fatalf("mulwide dsthi must be DX")
	}
	a.Printf("\tMUL%s %s\n", x86Suffix(a), src2)
}
