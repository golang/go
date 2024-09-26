// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"cmd/internal/obj"
	"fmt"
)

func init() {
	obj.RegisterRegister(obj.RBaseLOONG64, REG_LAST+1, rconv)
	obj.RegisterOpcode(obj.ABaseLoong64, Anames)
}

func rconv(r int) string {
	if r == 0 {
		return "NONE"
	}
	if r == REGG {
		// Special case.
		return "g"
	}

	if REG_R0 <= r && r <= REG_R31 {
		return fmt.Sprintf("R%d", r-REG_R0)
	}

	if REG_F0 <= r && r <= REG_F31 {
		return fmt.Sprintf("F%d", r-REG_F0)
	}

	if REG_FCSR0 <= r && r <= REG_FCSR31 {
		return fmt.Sprintf("FCSR%d", r-REG_FCSR0)
	}

	if REG_FCC0 <= r && r <= REG_FCC31 {
		return fmt.Sprintf("FCC%d", r-REG_FCC0)
	}

	if REG_V0 <= r && r <= REG_V31 {
		return fmt.Sprintf("V%d", r-REG_V0)
	}

	if REG_X0 <= r && r <= REG_X31 {
		return fmt.Sprintf("X%d", r-REG_X0)
	}

	return fmt.Sprintf("Rgok(%d)", r-obj.RBaseLOONG64)
}

func DRconv(a int) string {
	s := "C_??"
	if a >= C_NONE && a <= C_NCLASS {
		s = cnames0[a]
	}
	return s
}
