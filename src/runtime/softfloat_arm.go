// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Software floating point interpretation of ARM 7500 FP instructions.
// The interpretation is not bit compatible with the 7500.
// It uses true little-endian doubles, while the 7500 used mixed-endian.

package runtime

import "unsafe"

const (
	_CPSR    = 14
	_FLAGS_N = 1 << 31
	_FLAGS_Z = 1 << 30
	_FLAGS_C = 1 << 29
	_FLAGS_V = 1 << 28
)

var fptrace = 0

func fabort() {
	throw("unsupported floating point instruction")
}

func fputf(reg uint32, val uint32) {
	_g_ := getg()
	_g_.m.freglo[reg] = val
}

func fputd(reg uint32, val uint64) {
	_g_ := getg()
	_g_.m.freglo[reg] = uint32(val)
	_g_.m.freghi[reg] = uint32(val >> 32)
}

func fgetd(reg uint32) uint64 {
	_g_ := getg()
	return uint64(_g_.m.freglo[reg]) | uint64(_g_.m.freghi[reg])<<32
}

func fprintregs() {
	_g_ := getg()
	for i := range _g_.m.freglo {
		print("\tf", i, ":\t", hex(_g_.m.freghi[i]), " ", hex(_g_.m.freglo[i]), "\n")
	}
}

func fstatus(nan bool, cmp int32) uint32 {
	if nan {
		return _FLAGS_C | _FLAGS_V
	}
	if cmp == 0 {
		return _FLAGS_Z | _FLAGS_C
	}
	if cmp < 0 {
		return _FLAGS_N
	}
	return _FLAGS_C
}

// conditions array record the required CPSR cond field for the
// first 5 pairs of conditional execution opcodes
// higher 4 bits are must set, lower 4 bits are must clear
var conditions = [10 / 2]uint32{
	0 / 2: _FLAGS_Z>>24 | 0, // 0: EQ (Z set), 1: NE (Z clear)
	2 / 2: _FLAGS_C>>24 | 0, // 2: CS/HS (C set), 3: CC/LO (C clear)
	4 / 2: _FLAGS_N>>24 | 0, // 4: MI (N set), 5: PL (N clear)
	6 / 2: _FLAGS_V>>24 | 0, // 6: VS (V set), 7: VC (V clear)
	8 / 2: _FLAGS_C>>24 |
		_FLAGS_Z>>28,
}

const _FAULT = 0x80000000 // impossible PC offset

// returns number of words that the fp instruction
// is occupying, 0 if next instruction isn't float.
func stepflt(pc *uint32, regs *[15]uint32) uint32 {
	var i, opc, regd, regm, regn, cpsr uint32

	// m is locked in vlop_arm.s, so g.m cannot change during this function call,
	// so caching it in a local variable is safe.
	m := getg().m
	i = *pc

	if fptrace > 0 {
		print("stepflt ", pc, " ", hex(i), " (cpsr ", hex(regs[_CPSR]>>28), ")\n")
	}

	opc = i >> 28
	if opc == 14 { // common case first
		goto execute
	}

	cpsr = regs[_CPSR] >> 28
	switch opc {
	case 0, 1, 2, 3, 4, 5, 6, 7, 8, 9:
		if cpsr&(conditions[opc/2]>>4) == conditions[opc/2]>>4 &&
			cpsr&(conditions[opc/2]&0xf) == 0 {
			if opc&1 != 0 {
				return 1
			}
		} else {
			if opc&1 == 0 {
				return 1
			}
		}

	case 10, 11: // GE (N == V), LT (N != V)
		if cpsr&(_FLAGS_N>>28) == cpsr&(_FLAGS_V>>28) {
			if opc&1 != 0 {
				return 1
			}
		} else {
			if opc&1 == 0 {
				return 1
			}
		}

	case 12, 13: // GT (N == V and Z == 0), LE (N != V or Z == 1)
		if cpsr&(_FLAGS_N>>28) == cpsr&(_FLAGS_V>>28) &&
			cpsr&(_FLAGS_Z>>28) == 0 {
			if opc&1 != 0 {
				return 1
			}
		} else {
			if opc&1 == 0 {
				return 1
			}
		}

	case 14: // AL
		// ok

	case 15: // shouldn't happen
		return 0
	}

	if fptrace > 0 {
		print("conditional ", hex(opc), " (cpsr ", hex(cpsr), ") pass\n")
	}
	i = 0xe<<28 | i&(1<<28-1)

execute:
	// special cases
	if i&0xfffff000 == 0xe59fb000 {
		// load r11 from pc-relative address.
		// might be part of a floating point move
		// (or might not, but no harm in simulating
		// one instruction too many).
		addr := (*[1]uint32)(add(unsafe.Pointer(pc), uintptr(i&0xfff+8)))
		regs[11] = addr[0]

		if fptrace > 0 {
			print("*** cpu R[11] = *(", addr, ") ", hex(regs[11]), "\n")
		}
		return 1
	}
	if i == 0xe08fb00b {
		// add pc to r11
		// might be part of a PIC floating point move
		// (or might not, but again no harm done).
		regs[11] += uint32(uintptr(unsafe.Pointer(pc))) + 8

		if fptrace > 0 {
			print("*** cpu R[11] += pc ", hex(regs[11]), "\n")
		}
		return 1
	}
	if i&0xfffffff0 == 0xe08bb000 {
		r := i & 0xf
		// add r to r11.
		// might be part of a large offset address calculation
		// (or might not, but again no harm done).
		regs[11] += regs[r]

		if fptrace > 0 {
			print("*** cpu R[11] += R[", r, "] ", hex(regs[11]), "\n")
		}
		return 1
	}
	if i == 0xeef1fa10 {
		regs[_CPSR] = regs[_CPSR]&0x0fffffff | m.fflag

		if fptrace > 0 {
			print("*** fpsr R[CPSR] = F[CPSR] ", hex(regs[_CPSR]), "\n")
		}
		return 1
	}
	if i&0xff000000 == 0xea000000 {
		// unconditional branch
		// can happen in the middle of floating point
		// if the linker decides it is time to lay down
		// a sequence of instruction stream constants.
		delta := int32(i&0xffffff) << 8 >> 8 // sign extend

		if fptrace > 0 {
			print("*** cpu PC += ", hex((delta+2)*4), "\n")
		}
		return uint32(delta + 2)
	}

	// load/store regn is cpureg, regm is 8bit offset
	regd = i >> 12 & 0xf
	regn = i >> 16 & 0xf
	regm = i & 0xff << 2 // PLUS or MINUS ??

	switch i & 0xfff00f00 {
	case 0xed900a00: // single load
		uaddr := uintptr(regs[regn] + regm)
		if uaddr < 4096 {
			if fptrace > 0 {
				print("*** load @", hex(uaddr), " => fault\n")
			}
			return _FAULT
		}
		addr := (*[1]uint32)(unsafe.Pointer(uaddr))
		m.freglo[regd] = addr[0]

		if fptrace > 0 {
			print("*** load F[", regd, "] = ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xed900b00: // double load
		uaddr := uintptr(regs[regn] + regm)
		if uaddr < 4096 {
			if fptrace > 0 {
				print("*** double load @", hex(uaddr), " => fault\n")
			}
			return _FAULT
		}
		addr := (*[2]uint32)(unsafe.Pointer(uaddr))
		m.freglo[regd] = addr[0]
		m.freghi[regd] = addr[1]

		if fptrace > 0 {
			print("*** load D[", regd, "] = ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xed800a00: // single store
		uaddr := uintptr(regs[regn] + regm)
		if uaddr < 4096 {
			if fptrace > 0 {
				print("*** store @", hex(uaddr), " => fault\n")
			}
			return _FAULT
		}
		addr := (*[1]uint32)(unsafe.Pointer(uaddr))
		addr[0] = m.freglo[regd]

		if fptrace > 0 {
			print("*** *(", addr, ") = ", hex(addr[0]), "\n")
		}
		return 1

	case 0xed800b00: // double store
		uaddr := uintptr(regs[regn] + regm)
		if uaddr < 4096 {
			if fptrace > 0 {
				print("*** double store @", hex(uaddr), " => fault\n")
			}
			return _FAULT
		}
		addr := (*[2]uint32)(unsafe.Pointer(uaddr))
		addr[0] = m.freglo[regd]
		addr[1] = m.freghi[regd]

		if fptrace > 0 {
			print("*** *(", addr, ") = ", hex(addr[1]), "-", hex(addr[0]), "\n")
		}
		return 1
	}

	// regd, regm, regn are 4bit variables
	regm = i >> 0 & 0xf
	switch i & 0xfff00ff0 {
	case 0xf3000110: // veor
		m.freglo[regd] = m.freglo[regm] ^ m.freglo[regn]
		m.freghi[regd] = m.freghi[regm] ^ m.freghi[regn]

		if fptrace > 0 {
			print("*** veor D[", regd, "] = ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb00b00: // D[regd] = const(regn,regm)
		regn = regn<<4 | regm
		regm = 0x40000000
		if regn&0x80 != 0 {
			regm |= 0x80000000
		}
		if regn&0x40 != 0 {
			regm ^= 0x7fc00000
		}
		regm |= regn & 0x3f << 16
		m.freglo[regd] = 0
		m.freghi[regd] = regm

		if fptrace > 0 {
			print("*** immed D[", regd, "] = ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb00a00: // F[regd] = const(regn,regm)
		regn = regn<<4 | regm
		regm = 0x40000000
		if regn&0x80 != 0 {
			regm |= 0x80000000
		}
		if regn&0x40 != 0 {
			regm ^= 0x7e000000
		}
		regm |= regn & 0x3f << 19
		m.freglo[regd] = regm

		if fptrace > 0 {
			print("*** immed D[", regd, "] = ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee300b00: // D[regd] = D[regn]+D[regm]
		fputd(regd, fadd64(fgetd(regn), fgetd(regm)))

		if fptrace > 0 {
			print("*** add D[", regd, "] = D[", regn, "]+D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee300a00: // F[regd] = F[regn]+F[regm]
		m.freglo[regd] = f64to32(fadd64(f32to64(m.freglo[regn]), f32to64(m.freglo[regm])))

		if fptrace > 0 {
			print("*** add F[", regd, "] = F[", regn, "]+F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee300b40: // D[regd] = D[regn]-D[regm]
		fputd(regd, fsub64(fgetd(regn), fgetd(regm)))

		if fptrace > 0 {
			print("*** sub D[", regd, "] = D[", regn, "]-D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee300a40: // F[regd] = F[regn]-F[regm]
		m.freglo[regd] = f64to32(fsub64(f32to64(m.freglo[regn]), f32to64(m.freglo[regm])))

		if fptrace > 0 {
			print("*** sub F[", regd, "] = F[", regn, "]-F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee200b00: // D[regd] = D[regn]*D[regm]
		fputd(regd, fmul64(fgetd(regn), fgetd(regm)))

		if fptrace > 0 {
			print("*** mul D[", regd, "] = D[", regn, "]*D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee200a00: // F[regd] = F[regn]*F[regm]
		m.freglo[regd] = f64to32(fmul64(f32to64(m.freglo[regn]), f32to64(m.freglo[regm])))

		if fptrace > 0 {
			print("*** mul F[", regd, "] = F[", regn, "]*F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee800b00: // D[regd] = D[regn]/D[regm]
		fputd(regd, fdiv64(fgetd(regn), fgetd(regm)))

		if fptrace > 0 {
			print("*** div D[", regd, "] = D[", regn, "]/D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee800a00: // F[regd] = F[regn]/F[regm]
		m.freglo[regd] = f64to32(fdiv64(f32to64(m.freglo[regn]), f32to64(m.freglo[regm])))

		if fptrace > 0 {
			print("*** div F[", regd, "] = F[", regn, "]/F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xee000b10: // S[regn] = R[regd] (MOVW) (regm ignored)
		m.freglo[regn] = regs[regd]

		if fptrace > 0 {
			print("*** cpy S[", regn, "] = R[", regd, "] ", hex(m.freglo[regn]), "\n")
		}
		return 1

	case 0xee100b10: // R[regd] = S[regn] (MOVW) (regm ignored)
		regs[regd] = m.freglo[regn]

		if fptrace > 0 {
			print("*** cpy R[", regd, "] = S[", regn, "] ", hex(regs[regd]), "\n")
		}
		return 1
	}

	// regd, regm are 4bit variables
	switch i & 0xffff0ff0 {
	case 0xeeb00a40: // F[regd] = F[regm] (MOVF)
		m.freglo[regd] = m.freglo[regm]

		if fptrace > 0 {
			print("*** F[", regd, "] = F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb00b40: // D[regd] = D[regm] (MOVD)
		m.freglo[regd] = m.freglo[regm]
		m.freghi[regd] = m.freghi[regm]

		if fptrace > 0 {
			print("*** D[", regd, "] = D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb10bc0: // D[regd] = sqrt D[regm]
		fputd(regd, sqrt(fgetd(regm)))

		if fptrace > 0 {
			print("*** D[", regd, "] = sqrt D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb00bc0: // D[regd] = abs D[regm]
		m.freglo[regd] = m.freglo[regm]
		m.freghi[regd] = m.freghi[regm] & (1<<31 - 1)

		if fptrace > 0 {
			print("*** D[", regd, "] = abs D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb00ac0: // F[regd] = abs F[regm]
		m.freglo[regd] = m.freglo[regm] & (1<<31 - 1)

		if fptrace > 0 {
			print("*** F[", regd, "] = abs F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb40bc0: // D[regd] :: D[regm] (CMPD)
		cmp, nan := fcmp64(fgetd(regd), fgetd(regm))
		m.fflag = fstatus(nan, cmp)

		if fptrace > 0 {
			print("*** cmp D[", regd, "]::D[", regm, "] ", hex(m.fflag), "\n")
		}
		return 1

	case 0xeeb40ac0: // F[regd] :: F[regm] (CMPF)
		cmp, nan := fcmp64(f32to64(m.freglo[regd]), f32to64(m.freglo[regm]))
		m.fflag = fstatus(nan, cmp)

		if fptrace > 0 {
			print("*** cmp F[", regd, "]::F[", regm, "] ", hex(m.fflag), "\n")
		}
		return 1

	case 0xeeb70ac0: // D[regd] = F[regm] (MOVFD)
		fputd(regd, f32to64(m.freglo[regm]))

		if fptrace > 0 {
			print("*** f2d D[", regd, "]=F[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb70bc0: // F[regd] = D[regm] (MOVDF)
		m.freglo[regd] = f64to32(fgetd(regm))

		if fptrace > 0 {
			print("*** d2f F[", regd, "]=D[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeebd0ac0: // S[regd] = F[regm] (MOVFW)
		sval, ok := f64toint(f32to64(m.freglo[regm]))
		if !ok || int64(int32(sval)) != sval {
			sval = 0
		}
		m.freglo[regd] = uint32(sval)
		if fptrace > 0 {
			print("*** fix S[", regd, "]=F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeebc0ac0: // S[regd] = F[regm] (MOVFW.U)
		sval, ok := f64toint(f32to64(m.freglo[regm]))
		if !ok || int64(uint32(sval)) != sval {
			sval = 0
		}
		m.freglo[regd] = uint32(sval)

		if fptrace > 0 {
			print("*** fix unsigned S[", regd, "]=F[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeebd0bc0: // S[regd] = D[regm] (MOVDW)
		sval, ok := f64toint(fgetd(regm))
		if !ok || int64(int32(sval)) != sval {
			sval = 0
		}
		m.freglo[regd] = uint32(sval)

		if fptrace > 0 {
			print("*** fix S[", regd, "]=D[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeebc0bc0: // S[regd] = D[regm] (MOVDW.U)
		sval, ok := f64toint(fgetd(regm))
		if !ok || int64(uint32(sval)) != sval {
			sval = 0
		}
		m.freglo[regd] = uint32(sval)

		if fptrace > 0 {
			print("*** fix unsigned S[", regd, "]=D[", regm, "] ", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb80ac0: // D[regd] = S[regm] (MOVWF)
		cmp := int32(m.freglo[regm])
		if cmp < 0 {
			fputf(regd, f64to32(fintto64(-int64(cmp))))
			m.freglo[regd] ^= 0x80000000
		} else {
			fputf(regd, f64to32(fintto64(int64(cmp))))
		}

		if fptrace > 0 {
			print("*** float D[", regd, "]=S[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb80a40: // D[regd] = S[regm] (MOVWF.U)
		fputf(regd, f64to32(fintto64(int64(m.freglo[regm]))))

		if fptrace > 0 {
			print("*** float unsigned D[", regd, "]=S[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb80bc0: // D[regd] = S[regm] (MOVWD)
		cmp := int32(m.freglo[regm])
		if cmp < 0 {
			fputd(regd, fintto64(-int64(cmp)))
			m.freghi[regd] ^= 0x80000000
		} else {
			fputd(regd, fintto64(int64(cmp)))
		}

		if fptrace > 0 {
			print("*** float D[", regd, "]=S[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1

	case 0xeeb80b40: // D[regd] = S[regm] (MOVWD.U)
		fputd(regd, fintto64(int64(m.freglo[regm])))

		if fptrace > 0 {
			print("*** float unsigned D[", regd, "]=S[", regm, "] ", hex(m.freghi[regd]), "-", hex(m.freglo[regd]), "\n")
		}
		return 1
	}

	if i&0xff000000 == 0xee000000 || i&0xff000000 == 0xed000000 {
		print("stepflt ", pc, " ", hex(i), "\n")
		fabort()
	}
	return 0
}

//go:nosplit
func _sfloat2(pc uint32, regs [15]uint32) (newpc uint32) {
	systemstack(func() {
		newpc = sfloat2(pc, &regs)
	})
	return
}

func _sfloatpanic()

func sfloat2(pc uint32, regs *[15]uint32) uint32 {
	first := true
	for {
		skip := stepflt((*uint32)(unsafe.Pointer(uintptr(pc))), regs)
		if skip == 0 {
			break
		}
		first = false
		if skip == _FAULT {
			// Encountered bad address in store/load.
			// Record signal information and return to assembly
			// trampoline that fakes the call.
			const SIGSEGV = 11
			curg := getg().m.curg
			curg.sig = SIGSEGV
			curg.sigcode0 = 0
			curg.sigcode1 = 0
			curg.sigpc = uintptr(pc)
			pc = uint32(funcPC(_sfloatpanic))
			break
		}
		pc += 4 * skip
	}
	if first {
		print("sfloat2 ", pc, " ", hex(*(*uint32)(unsafe.Pointer(uintptr(pc)))), "\n")
		fabort() // not ok to fail first instruction
	}
	return pc
}
