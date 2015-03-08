// Derived from Inferno utils/6c/peep.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/peep.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

package main

import (
	"cmd/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"fmt"
)

var gactive uint32

func peep(firstp *obj.Prog) {
	// TODO(aram)
}

func excise(r *gc.Flow) {
	p := (*obj.Prog)(r.Prog)
	if gc.Debug['P'] != 0 && gc.Debug['v'] != 0 {
		fmt.Printf("%v ===delete===\n", p)
	}
	obj.Nopout(p)
	gc.Ostats.Ndelmov++
}

func regtyp(a *obj.Addr) bool {
	// TODO(rsc): Floating point register exclusions?
	return a.Type == obj.TYPE_REG && arm64.REG_R0 <= a.Reg && a.Reg <= arm64.REG_F31 && a.Reg != arm64.REGZERO
}

func sameaddr(a *obj.Addr, v *obj.Addr) bool {
	if a.Type != v.Type {
		return false
	}
	if regtyp(v) && a.Reg == v.Reg {
		return true
	}
	if v.Type == obj.NAME_AUTO || v.Type == obj.NAME_PARAM {
		if v.Offset == a.Offset {
			return true
		}
	}
	return false
}

func smallindir(a *obj.Addr, reg *obj.Addr) bool {
	return reg.Type == obj.TYPE_REG && a.Type == obj.TYPE_MEM && a.Reg == reg.Reg && 0 <= a.Offset && a.Offset < 4096
}

func stackaddr(a *obj.Addr) bool {
	return a.Type == obj.TYPE_REG && a.Reg == arm64.REGSP
}
