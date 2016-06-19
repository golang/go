// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"fmt"
	"strings"
)

// IntelSyntax returns the Intel assembler syntax for the instruction, as defined by Intel's XED tool.
func IntelSyntax(inst Inst) string {
	var iargs []Arg
	for _, a := range inst.Args {
		if a == nil {
			break
		}
		iargs = append(iargs, a)
	}

	switch inst.Op {
	case INSB, INSD, INSW, OUTSB, OUTSD, OUTSW, LOOPNE, JCXZ, JECXZ, JRCXZ, LOOP, LOOPE, MOV, XLATB:
		if inst.Op == MOV && (inst.Opcode>>16)&0xFFFC != 0x0F20 {
			break
		}
		for i, p := range inst.Prefix {
			if p&0xFF == PrefixAddrSize {
				inst.Prefix[i] &^= PrefixImplicit
			}
		}
	}

	switch inst.Op {
	case MOV:
		dst, _ := inst.Args[0].(Reg)
		src, _ := inst.Args[1].(Reg)
		if ES <= dst && dst <= GS && EAX <= src && src <= R15L {
			src -= EAX - AX
			iargs[1] = src
		}
		if ES <= dst && dst <= GS && RAX <= src && src <= R15 {
			src -= RAX - AX
			iargs[1] = src
		}

		if inst.Opcode>>24&^3 == 0xA0 {
			for i, p := range inst.Prefix {
				if p&0xFF == PrefixAddrSize {
					inst.Prefix[i] |= PrefixImplicit
				}
			}
		}
	}

	switch inst.Op {
	case AAM, AAD:
		if imm, ok := iargs[0].(Imm); ok {
			if inst.DataSize == 32 {
				iargs[0] = Imm(uint32(int8(imm)))
			} else if inst.DataSize == 16 {
				iargs[0] = Imm(uint16(int8(imm)))
			}
		}

	case PUSH:
		if imm, ok := iargs[0].(Imm); ok {
			iargs[0] = Imm(uint32(imm))
		}
	}

	for _, p := range inst.Prefix {
		if p&PrefixImplicit != 0 {
			for j, pj := range inst.Prefix {
				if pj&0xFF == p&0xFF {
					inst.Prefix[j] |= PrefixImplicit
				}
			}
		}
	}

	if inst.Op != 0 {
		for i, p := range inst.Prefix {
			switch p &^ PrefixIgnored {
			case PrefixData16, PrefixData32, PrefixCS, PrefixDS, PrefixES, PrefixSS:
				inst.Prefix[i] |= PrefixImplicit
			}
			if p.IsREX() {
				inst.Prefix[i] |= PrefixImplicit
			}
		}
	}

	if isLoop[inst.Op] || inst.Op == JCXZ || inst.Op == JECXZ || inst.Op == JRCXZ {
		for i, p := range inst.Prefix {
			if p == PrefixPT || p == PrefixPN {
				inst.Prefix[i] |= PrefixImplicit
			}
		}
	}

	switch inst.Op {
	case AAA, AAS, CBW, CDQE, CLC, CLD, CLI, CLTS, CMC, CPUID, CQO, CWD, DAA, DAS,
		FDECSTP, FINCSTP, FNCLEX, FNINIT, FNOP, FWAIT, HLT,
		ICEBP, INSB, INSD, INSW, INT, INTO, INVD, IRET, IRETQ,
		LAHF, LEAVE, LRET, MONITOR, MWAIT, NOP, OUTSB, OUTSD, OUTSW,
		PAUSE, POPA, POPF, POPFQ, PUSHA, PUSHF, PUSHFQ,
		RDMSR, RDPMC, RDTSC, RDTSCP, RET, RSM,
		SAHF, STC, STD, STI, SYSENTER, SYSEXIT, SYSRET,
		UD2, WBINVD, WRMSR, XEND, XLATB, XTEST:

		if inst.Op == NOP && inst.Opcode>>24 != 0x90 {
			break
		}
		if inst.Op == RET && inst.Opcode>>24 != 0xC3 {
			break
		}
		if inst.Op == INT && inst.Opcode>>24 != 0xCC {
			break
		}
		if inst.Op == LRET && inst.Opcode>>24 != 0xcb {
			break
		}
		for i, p := range inst.Prefix {
			if p&0xFF == PrefixDataSize {
				inst.Prefix[i] &^= PrefixImplicit | PrefixIgnored
			}
		}

	case 0:
		// ok
	}

	switch inst.Op {
	case INSB, INSD, INSW, OUTSB, OUTSD, OUTSW, MONITOR, MWAIT, XLATB:
		iargs = nil

	case STOSB, STOSW, STOSD, STOSQ:
		iargs = iargs[:1]

	case LODSB, LODSW, LODSD, LODSQ, SCASB, SCASW, SCASD, SCASQ:
		iargs = iargs[1:]
	}

	const (
		haveData16 = 1 << iota
		haveData32
		haveAddr16
		haveAddr32
		haveXacquire
		haveXrelease
		haveLock
		haveHintTaken
		haveHintNotTaken
		haveBnd
	)
	var prefixBits uint32
	prefix := ""
	for _, p := range inst.Prefix {
		if p == 0 {
			break
		}
		if p&0xFF == 0xF3 {
			prefixBits &^= haveBnd
		}
		if p&(PrefixImplicit|PrefixIgnored) != 0 {
			continue
		}
		switch p {
		default:
			prefix += strings.ToLower(p.String()) + " "
		case PrefixCS, PrefixDS, PrefixES, PrefixFS, PrefixGS, PrefixSS:
			if inst.Op == 0 {
				prefix += strings.ToLower(p.String()) + " "
			}
		case PrefixREPN:
			prefix += "repne "
		case PrefixLOCK:
			prefixBits |= haveLock
		case PrefixData16, PrefixDataSize:
			prefixBits |= haveData16
		case PrefixData32:
			prefixBits |= haveData32
		case PrefixAddrSize, PrefixAddr16:
			prefixBits |= haveAddr16
		case PrefixAddr32:
			prefixBits |= haveAddr32
		case PrefixXACQUIRE:
			prefixBits |= haveXacquire
		case PrefixXRELEASE:
			prefixBits |= haveXrelease
		case PrefixPT:
			prefixBits |= haveHintTaken
		case PrefixPN:
			prefixBits |= haveHintNotTaken
		case PrefixBND:
			prefixBits |= haveBnd
		}
	}
	switch inst.Op {
	case JMP:
		if inst.Opcode>>24 == 0xEB {
			prefixBits &^= haveBnd
		}
	case RET, LRET:
		prefixBits &^= haveData16 | haveData32
	}

	if prefixBits&haveXacquire != 0 {
		prefix += "xacquire "
	}
	if prefixBits&haveXrelease != 0 {
		prefix += "xrelease "
	}
	if prefixBits&haveLock != 0 {
		prefix += "lock "
	}
	if prefixBits&haveBnd != 0 {
		prefix += "bnd "
	}
	if prefixBits&haveHintTaken != 0 {
		prefix += "hint-taken "
	}
	if prefixBits&haveHintNotTaken != 0 {
		prefix += "hint-not-taken "
	}
	if prefixBits&haveAddr16 != 0 {
		prefix += "addr16 "
	}
	if prefixBits&haveAddr32 != 0 {
		prefix += "addr32 "
	}
	if prefixBits&haveData16 != 0 {
		prefix += "data16 "
	}
	if prefixBits&haveData32 != 0 {
		prefix += "data32 "
	}

	if inst.Op == 0 {
		if prefix == "" {
			return "<no instruction>"
		}
		return prefix[:len(prefix)-1]
	}

	var args []string
	for _, a := range iargs {
		if a == nil {
			break
		}
		args = append(args, intelArg(&inst, a))
	}

	var op string
	switch inst.Op {
	case NOP:
		if inst.Opcode>>24 == 0x0F {
			if inst.DataSize == 16 {
				args = append(args, "ax")
			} else {
				args = append(args, "eax")
			}
		}

	case BLENDVPD, BLENDVPS, PBLENDVB:
		args = args[:2]

	case INT:
		if inst.Opcode>>24 == 0xCC {
			args = nil
			op = "int3"
		}

	case LCALL, LJMP:
		if len(args) == 2 {
			args[0], args[1] = args[1], args[0]
		}

	case FCHS, FABS, FTST, FLDPI, FLDL2E, FLDLG2, F2XM1, FXAM, FLD1, FLDL2T, FSQRT, FRNDINT, FCOS, FSIN:
		if len(args) == 0 {
			args = append(args, "st0")
		}

	case FPTAN, FSINCOS, FUCOMPP, FCOMPP, FYL2X, FPATAN, FXTRACT, FPREM1, FPREM, FYL2XP1, FSCALE:
		if len(args) == 0 {
			args = []string{"st0", "st1"}
		}

	case FST, FSTP, FISTTP, FIST, FISTP, FBSTP:
		if len(args) == 1 {
			args = append(args, "st0")
		}

	case FLD, FXCH, FCOM, FCOMP, FIADD, FIMUL, FICOM, FICOMP, FISUBR, FIDIV, FUCOM, FUCOMP, FILD, FBLD, FADD, FMUL, FSUB, FSUBR, FISUB, FDIV, FDIVR, FIDIVR:
		if len(args) == 1 {
			args = []string{"st0", args[0]}
		}

	case MASKMOVDQU, MASKMOVQ, XLATB, OUTSB, OUTSW, OUTSD:
	FixSegment:
		for i := len(inst.Prefix) - 1; i >= 0; i-- {
			p := inst.Prefix[i] & 0xFF
			switch p {
			case PrefixCS, PrefixES, PrefixFS, PrefixGS, PrefixSS:
				if inst.Mode != 64 || p == PrefixFS || p == PrefixGS {
					args = append(args, strings.ToLower((inst.Prefix[i] & 0xFF).String()))
					break FixSegment
				}
			case PrefixDS:
				if inst.Mode != 64 {
					break FixSegment
				}
			}
		}
	}

	if op == "" {
		op = intelOp[inst.Op]
	}
	if op == "" {
		op = strings.ToLower(inst.Op.String())
	}
	if args != nil {
		op += " " + strings.Join(args, ", ")
	}
	return prefix + op
}

func intelArg(inst *Inst, arg Arg) string {
	switch a := arg.(type) {
	case Imm:
		if inst.Mode == 32 {
			return fmt.Sprintf("%#x", uint32(a))
		}
		if Imm(int32(a)) == a {
			return fmt.Sprintf("%#x", int64(a))
		}
		return fmt.Sprintf("%#x", uint64(a))
	case Mem:
		if a.Base == EIP {
			a.Base = RIP
		}
		prefix := ""
		switch inst.MemBytes {
		case 1:
			prefix = "byte "
		case 2:
			prefix = "word "
		case 4:
			prefix = "dword "
		case 8:
			prefix = "qword "
		case 16:
			prefix = "xmmword "
		}
		switch inst.Op {
		case INVLPG:
			prefix = "byte "
		case STOSB, MOVSB, CMPSB, LODSB, SCASB:
			prefix = "byte "
		case STOSW, MOVSW, CMPSW, LODSW, SCASW:
			prefix = "word "
		case STOSD, MOVSD, CMPSD, LODSD, SCASD:
			prefix = "dword "
		case STOSQ, MOVSQ, CMPSQ, LODSQ, SCASQ:
			prefix = "qword "
		case LAR:
			prefix = "word "
		case BOUND:
			if inst.Mode == 32 {
				prefix = "qword "
			} else {
				prefix = "dword "
			}
		case PREFETCHW, PREFETCHNTA, PREFETCHT0, PREFETCHT1, PREFETCHT2, CLFLUSH:
			prefix = "zmmword "
		}
		switch inst.Op {
		case MOVSB, MOVSW, MOVSD, MOVSQ, CMPSB, CMPSW, CMPSD, CMPSQ, STOSB, STOSW, STOSD, STOSQ, SCASB, SCASW, SCASD, SCASQ, LODSB, LODSW, LODSD, LODSQ:
			switch a.Base {
			case DI, EDI, RDI:
				if a.Segment == ES {
					a.Segment = 0
				}
			case SI, ESI, RSI:
				if a.Segment == DS {
					a.Segment = 0
				}
			}
		case LEA:
			a.Segment = 0
		default:
			switch a.Base {
			case SP, ESP, RSP, BP, EBP, RBP:
				if a.Segment == SS {
					a.Segment = 0
				}
			default:
				if a.Segment == DS {
					a.Segment = 0
				}
			}
		}

		if inst.Mode == 64 && a.Segment != FS && a.Segment != GS {
			a.Segment = 0
		}

		prefix += "ptr "
		if a.Segment != 0 {
			prefix += strings.ToLower(a.Segment.String()) + ":"
		}
		prefix += "["
		if a.Base != 0 {
			prefix += intelArg(inst, a.Base)
		}
		if a.Scale != 0 && a.Index != 0 {
			if a.Base != 0 {
				prefix += "+"
			}
			prefix += fmt.Sprintf("%s*%d", intelArg(inst, a.Index), a.Scale)
		}
		if a.Disp != 0 {
			if prefix[len(prefix)-1] == '[' && (a.Disp >= 0 || int64(int32(a.Disp)) != a.Disp) {
				prefix += fmt.Sprintf("%#x", uint64(a.Disp))
			} else {
				prefix += fmt.Sprintf("%+#x", a.Disp)
			}
		}
		prefix += "]"
		return prefix
	case Rel:
		return fmt.Sprintf(".%+#x", int64(a))
	case Reg:
		if int(a) < len(intelReg) && intelReg[a] != "" {
			return intelReg[a]
		}
	}
	return strings.ToLower(arg.String())
}

var intelOp = map[Op]string{
	JAE:       "jnb",
	JA:        "jnbe",
	JGE:       "jnl",
	JNE:       "jnz",
	JG:        "jnle",
	JE:        "jz",
	SETAE:     "setnb",
	SETA:      "setnbe",
	SETGE:     "setnl",
	SETNE:     "setnz",
	SETG:      "setnle",
	SETE:      "setz",
	CMOVAE:    "cmovnb",
	CMOVA:     "cmovnbe",
	CMOVGE:    "cmovnl",
	CMOVNE:    "cmovnz",
	CMOVG:     "cmovnle",
	CMOVE:     "cmovz",
	LCALL:     "call far",
	LJMP:      "jmp far",
	LRET:      "ret far",
	ICEBP:     "int1",
	MOVSD_XMM: "movsd",
	XLATB:     "xlat",
}

var intelReg = [...]string{
	F0:  "st0",
	F1:  "st1",
	F2:  "st2",
	F3:  "st3",
	F4:  "st4",
	F5:  "st5",
	F6:  "st6",
	F7:  "st7",
	M0:  "mmx0",
	M1:  "mmx1",
	M2:  "mmx2",
	M3:  "mmx3",
	M4:  "mmx4",
	M5:  "mmx5",
	M6:  "mmx6",
	M7:  "mmx7",
	X0:  "xmm0",
	X1:  "xmm1",
	X2:  "xmm2",
	X3:  "xmm3",
	X4:  "xmm4",
	X5:  "xmm5",
	X6:  "xmm6",
	X7:  "xmm7",
	X8:  "xmm8",
	X9:  "xmm9",
	X10: "xmm10",
	X11: "xmm11",
	X12: "xmm12",
	X13: "xmm13",
	X14: "xmm14",
	X15: "xmm15",

	// TODO: Maybe the constants are named wrong.
	SPB: "spl",
	BPB: "bpl",
	SIB: "sil",
	DIB: "dil",

	R8L:  "r8d",
	R9L:  "r9d",
	R10L: "r10d",
	R11L: "r11d",
	R12L: "r12d",
	R13L: "r13d",
	R14L: "r14d",
	R15L: "r15d",
}
