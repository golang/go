// Inferno utils/6c/list.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6c/list.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
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

package x86

import (
	"cmd/internal/obj"
	"fmt"
)

var Register = []string{
	"AL", // [D_AL]
	"CL",
	"DL",
	"BL",
	"SPB",
	"BPB",
	"SIB",
	"DIB",
	"R8B",
	"R9B",
	"R10B",
	"R11B",
	"R12B",
	"R13B",
	"R14B",
	"R15B",
	"AX", // [D_AX]
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"R14",
	"R15",
	"AH",
	"CH",
	"DH",
	"BH",
	"F0", // [D_F0]
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"M0",
	"M1",
	"M2",
	"M3",
	"M4",
	"M5",
	"M6",
	"M7",
	"K0",
	"K1",
	"K2",
	"K3",
	"K4",
	"K5",
	"K6",
	"K7",
	"X0",
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",
	"X8",
	"X9",
	"X10",
	"X11",
	"X12",
	"X13",
	"X14",
	"X15",
	"X16",
	"X17",
	"X18",
	"X19",
	"X20",
	"X21",
	"X22",
	"X23",
	"X24",
	"X25",
	"X26",
	"X27",
	"X28",
	"X29",
	"X30",
	"X31",
	"Y0",
	"Y1",
	"Y2",
	"Y3",
	"Y4",
	"Y5",
	"Y6",
	"Y7",
	"Y8",
	"Y9",
	"Y10",
	"Y11",
	"Y12",
	"Y13",
	"Y14",
	"Y15",
	"Y16",
	"Y17",
	"Y18",
	"Y19",
	"Y20",
	"Y21",
	"Y22",
	"Y23",
	"Y24",
	"Y25",
	"Y26",
	"Y27",
	"Y28",
	"Y29",
	"Y30",
	"Y31",
	"Z0",
	"Z1",
	"Z2",
	"Z3",
	"Z4",
	"Z5",
	"Z6",
	"Z7",
	"Z8",
	"Z9",
	"Z10",
	"Z11",
	"Z12",
	"Z13",
	"Z14",
	"Z15",
	"Z16",
	"Z17",
	"Z18",
	"Z19",
	"Z20",
	"Z21",
	"Z22",
	"Z23",
	"Z24",
	"Z25",
	"Z26",
	"Z27",
	"Z28",
	"Z29",
	"Z30",
	"Z31",
	"CS", // [D_CS]
	"SS",
	"DS",
	"ES",
	"FS",
	"GS",
	"GDTR", // [D_GDTR]
	"IDTR", // [D_IDTR]
	"LDTR", // [D_LDTR]
	"MSW",  // [D_MSW]
	"TASK", // [D_TASK]
	"CR0",  // [D_CR]
	"CR1",
	"CR2",
	"CR3",
	"CR4",
	"CR5",
	"CR6",
	"CR7",
	"CR8",
	"CR9",
	"CR10",
	"CR11",
	"CR12",
	"CR13",
	"CR14",
	"CR15",
	"DR0", // [D_DR]
	"DR1",
	"DR2",
	"DR3",
	"DR4",
	"DR5",
	"DR6",
	"DR7",
	"TR0", // [D_TR]
	"TR1",
	"TR2",
	"TR3",
	"TR4",
	"TR5",
	"TR6",
	"TR7",
	"TLS",    // [D_TLS]
	"MAXREG", // [MAXREG]
}

func init() {
	obj.RegisterRegister(REG_AL, REG_AL+len(Register), rconv)
	obj.RegisterOpcode(obj.ABaseAMD64, Anames)
	obj.RegisterRegisterList(obj.RegListX86Lo, obj.RegListX86Hi, rlconv)
	obj.RegisterOpSuffix("386", opSuffixString)
	obj.RegisterOpSuffix("amd64", opSuffixString)
}

func rconv(r int) string {
	if REG_AL <= r && r-REG_AL < len(Register) {
		return Register[r-REG_AL]
	}
	return fmt.Sprintf("Rgok(%d)", r-obj.RBaseAMD64)
}

func rlconv(bits int64) string {
	reg0, reg1 := decodeRegisterRange(bits)
	return fmt.Sprintf("[%s-%s]", rconv(reg0), rconv(reg1))
}

func opSuffixString(s uint8) string {
	return "." + opSuffix(s).String()
}
