// Inferno utils/5c/reg.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/5c/reg.c
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

package arm

import "cmd/internal/obj/arm"

/*
 *	bit	reg
 *	0	R0
 *	1	R1
 *	...	...
 *	10	R10
 *	12  R12
 *
 *	bit	reg
 *	18	F2
 *	19	F3
 *	...	...
 *	31	F15
 */
func RtoB(r int) uint64 {
	if arm.REG_R0 <= r && r <= arm.REG_R15 {
		if r >= arm.REGTMP-2 && r != arm.REG_R12 { // excluded R9 and R10 for m and g, but not R12
			return 0
		}
		return 1 << uint(r-arm.REG_R0)
	}

	if arm.REG_F0 <= r && r <= arm.REG_F15 {
		if r < arm.REG_F2 || r > arm.REG_F0+arm.NFREG-1 {
			return 0
		}
		return 1 << uint((r-arm.REG_F0)+16)
	}

	return 0
}
