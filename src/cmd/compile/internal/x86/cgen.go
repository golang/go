// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

/*
 * generate an addressable node in res, containing the value of n.
 * n is an array index, and might be any size; res width is <= 32-bit.
 * returns Prog* to patch to panic call.
 */
func igenindex(n *gc.Node, res *gc.Node, bounded bool) *obj.Prog {
	if !gc.Is64(n.Type) {
		if n.Addable && (gc.Simtype[n.Etype] == gc.TUINT32 || gc.Simtype[n.Etype] == gc.TINT32) {
			// nothing to do.
			*res = *n
		} else {
			gc.Tempname(res, gc.Types[gc.TUINT32])
			gc.Cgen(n, res)
		}

		return nil
	}

	var tmp gc.Node
	gc.Tempname(&tmp, gc.Types[gc.TINT64])
	gc.Cgen(n, &tmp)
	var lo gc.Node
	var hi gc.Node
	split64(&tmp, &lo, &hi)
	gc.Tempname(res, gc.Types[gc.TUINT32])
	gmove(&lo, res)
	if bounded {
		splitclean()
		return nil
	}

	var zero gc.Node
	gc.Nodconst(&zero, gc.Types[gc.TINT32], 0)
	gins(x86.ACMPL, &hi, &zero)
	splitclean()
	return gc.Gbranch(x86.AJNE, nil, +1)
}

func blockcopy(n, res *gc.Node, osrc, odst, w int64) {
	var dst gc.Node
	gc.Nodreg(&dst, gc.Types[gc.Tptr], x86.REG_DI)
	var src gc.Node
	gc.Nodreg(&src, gc.Types[gc.Tptr], x86.REG_SI)

	var tsrc gc.Node
	gc.Tempname(&tsrc, gc.Types[gc.Tptr])
	var tdst gc.Node
	gc.Tempname(&tdst, gc.Types[gc.Tptr])
	if !n.Addable {
		gc.Agen(n, &tsrc)
	}
	if !res.Addable {
		gc.Agen(res, &tdst)
	}
	if n.Addable {
		gc.Agen(n, &src)
	} else {
		gmove(&tsrc, &src)
	}

	if res.Op == gc.ONAME {
		gc.Gvardef(res)
	}

	if res.Addable {
		gc.Agen(res, &dst)
	} else {
		gmove(&tdst, &dst)
	}

	c := int32(w % 4) // bytes
	q := int32(w / 4) // doublewords

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	if osrc < odst && odst < osrc+w {
		// reverse direction
		gins(x86.ASTD, nil, nil) // set direction flag
		if c > 0 {
			gconreg(x86.AADDL, w-1, x86.REG_SI)
			gconreg(x86.AADDL, w-1, x86.REG_DI)

			gconreg(x86.AMOVL, int64(c), x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSB, nil, nil) // MOVB *(SI)-,*(DI)-
		}

		if q > 0 {
			if c > 0 {
				gconreg(x86.AADDL, -3, x86.REG_SI)
				gconreg(x86.AADDL, -3, x86.REG_DI)
			} else {
				gconreg(x86.AADDL, w-4, x86.REG_SI)
				gconreg(x86.AADDL, w-4, x86.REG_DI)
			}

			gconreg(x86.AMOVL, int64(q), x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSL, nil, nil) // MOVL *(SI)-,*(DI)-
		}

		// we leave with the flag clear
		gins(x86.ACLD, nil, nil)
	} else {
		gins(x86.ACLD, nil, nil) // paranoia.  TODO(rsc): remove?

		// normal direction
		if q > 128 || (q >= 4 && gc.Nacl) {
			gconreg(x86.AMOVL, int64(q), x86.REG_CX)
			gins(x86.AREP, nil, nil)   // repeat
			gins(x86.AMOVSL, nil, nil) // MOVL *(SI)+,*(DI)+
		} else if q >= 4 {
			p := gins(obj.ADUFFCOPY, nil, nil)
			p.To.Type = obj.TYPE_ADDR
			p.To.Sym = gc.Linksym(gc.Pkglookup("duffcopy", gc.Runtimepkg))

			// 10 and 128 = magic constants: see ../../runtime/asm_386.s
			p.To.Offset = 10 * (128 - int64(q))
		} else if !gc.Nacl && c == 0 {
			var cx gc.Node
			gc.Nodreg(&cx, gc.Types[gc.TINT32], x86.REG_CX)

			// We don't need the MOVSL side-effect of updating SI and DI,
			// and issuing a sequence of MOVLs directly is faster.
			src.Op = gc.OINDREG

			dst.Op = gc.OINDREG
			for q > 0 {
				gmove(&src, &cx) // MOVL x+(SI),CX
				gmove(&cx, &dst) // MOVL CX,x+(DI)
				src.Xoffset += 4
				dst.Xoffset += 4
				q--
			}
		} else {
			for q > 0 {
				gins(x86.AMOVSL, nil, nil) // MOVL *(SI)+,*(DI)+
				q--
			}
		}

		for c > 0 {
			gins(x86.AMOVSB, nil, nil) // MOVB *(SI)+,*(DI)+
			c--
		}
	}
}
