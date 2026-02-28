// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

// This file contains some utility functions that can be used to decode
// vector instructions into both gnu and plan9 assembly.

func implicitMask(instOp Op) bool {
	switch instOp {
	case VADC_VIM, VADC_VVM, VADC_VXM, VFMERGE_VFM, VMADC_VIM, VMADC_VVM,
		VMADC_VXM, VMERGE_VIM, VMERGE_VVM, VMERGE_VXM, VMSBC_VVM, VMSBC_VXM,
		VSBC_VVM, VSBC_VXM:
		return true

	default:
		return false
	}
}

func imaOrFma(instOp Op) bool {
	switch instOp {
	case VFMACC_VF, VFMACC_VV, VFMADD_VF, VFMADD_VV, VFMSAC_VF, VFMSAC_VV,
		VFMSUB_VF, VFMSUB_VV, VFNMACC_VF, VFNMACC_VV, VFNMADD_VF, VFNMADD_VV,
		VFNMSAC_VF, VFNMSAC_VV, VFNMSUB_VF, VFNMSUB_VV, VFWMACC_VF, VFWMACC_VV,
		VFWMSAC_VF, VFWMSAC_VV, VFWNMACC_VF, VFWNMACC_VV, VFWNMSAC_VF,
		VFWNMSAC_VV, VMACC_VV, VMACC_VX, VMADD_VV, VMADD_VX, VNMSAC_VV,
		VNMSAC_VX, VNMSUB_VV, VNMSUB_VX, VWMACCSU_VV, VWMACCSU_VX, VWMACCUS_VX,
		VWMACCU_VV, VWMACCU_VX, VWMACC_VV, VWMACC_VX:
		return true

	default:
		return false
	}
}

func pseudoRVVLoad(instOp Op) string {
	switch instOp {
	case VL1RE8_V:
		return "VL1R.V"

	case VL2RE8_V:
		return "VL2R.V"

	case VL4RE8_V:
		return "VL4R.V"

	case VL8RE8_V:
		return "VL8R.V"
	}

	return ""
}

func pseudoRVVArith(instOp Op, rawArgs []Arg, args []string) (string, []string) {
	var op string

	switch instOp {
	case VRSUB_VX:
		if v, ok := rawArgs[1].(Reg); ok && v == X0 {
			op = "VNEG.V"
			args = append(args[:1], args[2:]...)
		}

	case VWADD_VX:
		if v, ok := rawArgs[1].(Reg); ok && v == X0 {
			op = "VWCVT.X.X.V"
			args = append(args[:1], args[2:]...)
		}

	case VWADDU_VX:
		if v, ok := rawArgs[1].(Reg); ok && v == X0 {
			op = "VWCVTU.X.X.V"
			args = append(args[:1], args[2:]...)
		}

	case VXOR_VI:
		if v, ok := rawArgs[1].(Simm); ok && v.Imm == -1 {
			op = "VNOT.V"
			args = append(args[:1], args[2:]...)
		}

	case VNSRL_WX:
		if v, ok := rawArgs[1].(Reg); ok && v == X0 {
			op = "VNCVT.X.X.W"
			args = append(args[:1], args[2:]...)
		}

	case VFSGNJN_VV:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		if ok1 && ok2 && vs1 == vs2 {
			op = "VFNEG.V"
			args = args[1:]
		}

	case VFSGNJX_VV:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		if ok1 && ok2 && vs1 == vs2 {
			op = "VFABS.V"
			args = args[1:]
		}

	case VMAND_MM:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		if ok1 && ok2 && vs1 == vs2 {
			op = "VMMV.M"
			args = args[1:]
		}

	case VMXOR_MM:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		vd, ok3 := rawArgs[2].(Reg)
		if ok1 && ok2 && ok3 && vs1 == vs2 && vd == vs1 {
			op = "VMCLR.M"
			args = args[2:]
		}

	case VMXNOR_MM:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		vd, ok3 := rawArgs[2].(Reg)
		if ok1 && ok2 && ok3 && vs1 == vs2 && vd == vs1 {
			op = "VMSET.M"
			args = args[2:]
		}

	case VMNAND_MM:
		vs2, ok1 := rawArgs[0].(Reg)
		vs1, ok2 := rawArgs[1].(Reg)
		if ok1 && ok2 && vs1 == vs2 {
			op = "VMNOT.M"
			args = args[1:]
		}
	}

	return op, args
}
