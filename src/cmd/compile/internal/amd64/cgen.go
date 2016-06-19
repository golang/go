// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

func blockcopy(n, ns *gc.Node, osrc, odst, w int64) {
	var noddi gc.Node
	gc.Nodreg(&noddi, gc.Types[gc.Tptr], x86.REG_DI)
	var nodsi gc.Node
	gc.Nodreg(&nodsi, gc.Types[gc.Tptr], x86.REG_SI)

	var nodl gc.Node
	var nodr gc.Node
	if n.Ullman >= ns.Ullman {
		gc.Agenr(n, &nodr, &nodsi)
		if ns.Op == gc.ONAME {
			gc.Gvardef(ns)
		}
		gc.Agenr(ns, &nodl, &noddi)
	} else {
		if ns.Op == gc.ONAME {
			gc.Gvardef(ns)
		}
		gc.Agenr(ns, &nodl, &noddi)
		gc.Agenr(n, &nodr, &nodsi)
	}

	if nodl.Reg != x86.REG_DI {
		gmove(&nodl, &noddi)
	}
	if nodr.Reg != x86.REG_SI {
		gmove(&nodr, &nodsi)
	}
	gc.Regfree(&nodl)
	gc.Regfree(&nodr)

	c := w % 8 // bytes
	q := w / 8 // quads

	var oldcx gc.Node
	var cx gc.Node
	savex(x86.REG_CX, &cx, &oldcx, nil, gc.Types[gc.TINT64])

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if osrc < odst && odst < osrc+w {
		// reverse direction
		gins(x86.ASTD, nil, nil) // set direction flag
		if c > 0 {
			gconreg(addptr, w-1, x86.REG_SI)
			gconreg(addptr, w-1, x86.REG_DI)

			gconreg(movptr, c, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSB, nil, nil) // MOVB *(SI)-,*(DI)-
		}

		if q > 0 {
			if c > 0 {
				gconreg(addptr, -7, x86.REG_SI)
				gconreg(addptr, -7, x86.REG_DI)
			} else {
				gconreg(addptr, w-8, x86.REG_SI)
				gconreg(addptr, w-8, x86.REG_DI)
			}

			gconreg(movptr, q, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)-,*(DI)-
		}

		// we leave with the flag clear
		gins(x86.ACLD, nil, nil)
	} else {
		// normal direction
		if q > 128 || (gc.Nacl && q >= 4) || (obj.Getgoos() == "plan9" && q >= 4) {
			gconreg(movptr, q, x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)+,*(DI)+
		} else if q >= 4 {
			var oldx0 gc.Node
			var x0 gc.Node
			savex(x86.REG_X0, &x0, &oldx0, nil, gc.Types[gc.TFLOAT64])

			p := gins(obj.ADUFFCOPY, nil, nil)
			p.To.Type = obj.TYPE_ADDR
			p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))

			// 64 blocks taking 14 bytes each
			// see ../../../../runtime/mkduff.go
			p.To.Offset = 14 * (64 - q/2)
			restx(&x0, &oldx0)

			if q%2 != 0 {
				gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)+,*(DI)+
			}
		} else if !gc.Nacl && c == 0 {
			// We don't need the MOVSQ side-effect of updating SI and DI,
			// and issuing a sequence of MOVQs directly is faster.
			nodsi.Op = gc.OINDREG

			noddi.Op = gc.OINDREG
			for q > 0 {
				gmove(&nodsi, &cx) // MOVQ x+(SI),CX
				gmove(&cx, &noddi) // MOVQ CX,x+(DI)
				nodsi.Xoffset += 8
				noddi.Xoffset += 8
				q--
			}
		} else {
			for q > 0 {
				gins(x86.AMOVSQ, nil, nil) // MOVQ *(SI)+,*(DI)+
				q--
			}
		}

		// copy the remaining c bytes
		if w < 4 || c <= 1 || (odst < osrc && osrc < odst+w) {
			for c > 0 {
				gins(x86.AMOVSB, nil, nil) // MOVB *(SI)+,*(DI)+
				c--
			}
		} else if w < 8 || c <= 4 {
			nodsi.Op = gc.OINDREG
			noddi.Op = gc.OINDREG
			cx.Type = gc.Types[gc.TINT32]
			nodsi.Type = gc.Types[gc.TINT32]
			noddi.Type = gc.Types[gc.TINT32]
			if c > 4 {
				nodsi.Xoffset = 0
				noddi.Xoffset = 0
				gmove(&nodsi, &cx)
				gmove(&cx, &noddi)
			}

			nodsi.Xoffset = c - 4
			noddi.Xoffset = c - 4
			gmove(&nodsi, &cx)
			gmove(&cx, &noddi)
		} else {
			nodsi.Op = gc.OINDREG
			noddi.Op = gc.OINDREG
			cx.Type = gc.Types[gc.TINT64]
			nodsi.Type = gc.Types[gc.TINT64]
			noddi.Type = gc.Types[gc.TINT64]
			nodsi.Xoffset = c - 8
			noddi.Xoffset = c - 8
			gmove(&nodsi, &cx)
			gmove(&cx, &noddi)
		}
	}

	restx(&cx, &oldcx)
}
