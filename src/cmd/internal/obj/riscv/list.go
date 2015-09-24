//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package riscv

import (
	"fmt"

	"cmd/internal/obj"
)

var (
	// Registers is a map of register names to integer IDs.
	Registers = make(map[string]int16)
	regNames = make(map[int16]string)

	// Instructions is a map of instruction names to integer IDs.
	Instructions = make(map[string]int)
)

func initRegisters() {
	for i := REG_X0; i <= REG_X31; i++ {
		name := fmt.Sprintf("X%d", i - REG_X0)
		Registers[name] = int16(i)
	}
	for i := REG_F0; i <= REG_F31; i++ {
		name := fmt.Sprintf("F%d", i - REG_F0)
		Registers[name] = int16(i)
	}
}

// TODO(myenik) Move to ABI names instead of x-names.
func initRegNames() {
	for name, reg := range Registers {
		regNames[reg] = name
	}

	// Special register names are reassigned for clarity.
	regNames[0] = "NONE"
	regNames[REG_G] = "G"
	regNames[REG_SP] = "SP"
	regNames[REG_FP] = "FP"
	regNames[REG_RA] = "RA"
	regNames[REG_ZERO] = "ZERO"
}

func initInstructions() {
	for i, s := range obj.Anames {
		Instructions[s] = i
	}
	for i, s := range Anames {
		if i >= obj.A_ARCHSPECIFIC {
			Instructions[s] = i + obj.ABaseRISCV
		}
	}
}

func init() {
	// initRegnames uses Registers during initialization,
	// and must be called after initRegisters.
	initRegisters()
	initRegNames()
	initInstructions()
	obj.RegisterRegister(obj.RBaseRISCV, REG_END, PrettyPrintReg)
	obj.RegisterOpcode(obj.ABaseRISCV, Anames)
}

func PrettyPrintReg(r int) string {
	name, ok := regNames[int16(r)]
	if !ok {
		name = fmt.Sprintf("R???%d", r) // Similar format to Aconv.
	}

	return name
}
