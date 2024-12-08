// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv

import (
	"fmt"

	"cmd/internal/obj"
)

func init() {
	obj.RegisterRegister(obj.RBaseRISCV, REG_END, RegName)
	obj.RegisterOpcode(obj.ABaseRISCV, Anames)
	obj.RegisterOpSuffix("riscv64", opSuffixString)
}

func RegName(r int) string {
	switch {
	case r == 0:
		return "NONE"
	case r == REG_G:
		return "g"
	case r == REG_SP:
		return "SP"
	case REG_X0 <= r && r <= REG_X31:
		return fmt.Sprintf("X%d", r-REG_X0)
	case REG_F0 <= r && r <= REG_F31:
		return fmt.Sprintf("F%d", r-REG_F0)
	case REG_V0 <= r && r <= REG_V31:
		return fmt.Sprintf("V%d", r-REG_V0)
	default:
		return fmt.Sprintf("Rgok(%d)", r-obj.RBaseRISCV)
	}
}

func opSuffixString(s uint8) string {
	if s&rmSuffixBit == 0 {
		return ""
	}

	ss, err := rmSuffixString(s)
	if err != nil {
		ss = fmt.Sprintf("<invalid 0x%x>", s)
	}
	if ss == "" {
		return ss
	}
	return fmt.Sprintf(".%s", ss)
}
