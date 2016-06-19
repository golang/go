// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
)

type direction int

const (
	_FORWARDS direction = iota
	_BACKWARDS
)

// blockcopy copies w bytes from &n to &res
func blockcopy(n, res *gc.Node, osrc, odst, w int64) {
	var dst gc.Node
	var src gc.Node
	if n.Ullman >= res.Ullman {
		gc.Agenr(n, &dst, res) // temporarily use dst
		gc.Regalloc(&src, gc.Types[gc.Tptr], nil)
		gins(s390x.AMOVD, &dst, &src)
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agen(res, &dst)
	} else {
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agenr(res, &dst, res)
		gc.Agenr(n, &src, nil)
	}
	defer gc.Regfree(&src)
	defer gc.Regfree(&dst)

	var tmp gc.Node
	gc.Regalloc(&tmp, gc.Types[gc.Tptr], nil)
	defer gc.Regfree(&tmp)

	offset := int64(0)
	dir := _FORWARDS
	if osrc < odst && odst < osrc+w {
		// Reverse. Can't use MVC, fall back onto basic moves.
		dir = _BACKWARDS
		const copiesPerIter = 2
		if w >= 8*copiesPerIter {
			cnt := w - (w % (8 * copiesPerIter))
			ginscon(s390x.AADD, w, &src)
			ginscon(s390x.AADD, w, &dst)

			var end gc.Node
			gc.Regalloc(&end, gc.Types[gc.Tptr], nil)
			p := gins(s390x.ASUB, nil, &end)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = cnt
			p.Reg = src.Reg

			var label *obj.Prog
			for i := 0; i < copiesPerIter; i++ {
				offset := int64(-8 * (i + 1))
				p := gins(s390x.AMOVD, &src, &tmp)
				p.From.Type = obj.TYPE_MEM
				p.From.Offset = offset
				if i == 0 {
					label = p
				}
				p = gins(s390x.AMOVD, &tmp, &dst)
				p.To.Type = obj.TYPE_MEM
				p.To.Offset = offset
			}

			ginscon(s390x.ASUB, 8*copiesPerIter, &src)
			ginscon(s390x.ASUB, 8*copiesPerIter, &dst)
			gins(s390x.ACMP, &src, &end)
			gc.Patch(gc.Gbranch(s390x.ABNE, nil, 0), label)
			gc.Regfree(&end)

			w -= cnt
		} else {
			offset = w
		}
	}

	if dir == _FORWARDS && w > 1024 {
		// Loop over MVCs
		cnt := w - (w % 256)

		var end gc.Node
		gc.Regalloc(&end, gc.Types[gc.Tptr], nil)
		add := gins(s390x.AADD, nil, &end)
		add.From.Type = obj.TYPE_CONST
		add.From.Offset = cnt
		add.Reg = src.Reg

		mvc := gins(s390x.AMVC, &src, &dst)
		mvc.From.Type = obj.TYPE_MEM
		mvc.From.Offset = 0
		mvc.To.Type = obj.TYPE_MEM
		mvc.To.Offset = 0
		mvc.From3 = new(obj.Addr)
		mvc.From3.Type = obj.TYPE_CONST
		mvc.From3.Offset = 256

		ginscon(s390x.AADD, 256, &src)
		ginscon(s390x.AADD, 256, &dst)
		gins(s390x.ACMP, &src, &end)
		gc.Patch(gc.Gbranch(s390x.ABNE, nil, 0), mvc)
		gc.Regfree(&end)

		w -= cnt
	}

	for w > 0 {
		cnt := w
		// If in reverse we can only do 8, 4, 2 or 1 bytes at a time.
		if dir == _BACKWARDS {
			switch {
			case cnt >= 8:
				cnt = 8
			case cnt >= 4:
				cnt = 4
			case cnt >= 2:
				cnt = 2
			}
		} else if cnt > 256 {
			cnt = 256
		}

		switch cnt {
		case 8, 4, 2, 1:
			op := s390x.AMOVB
			switch cnt {
			case 8:
				op = s390x.AMOVD
			case 4:
				op = s390x.AMOVW
			case 2:
				op = s390x.AMOVH
			}
			load := gins(op, &src, &tmp)
			load.From.Type = obj.TYPE_MEM
			load.From.Offset = offset

			store := gins(op, &tmp, &dst)
			store.To.Type = obj.TYPE_MEM
			store.To.Offset = offset

			if dir == _BACKWARDS {
				load.From.Offset -= cnt
				store.To.Offset -= cnt
			}

		default:
			p := gins(s390x.AMVC, &src, &dst)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = offset
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = offset
			p.From3 = new(obj.Addr)
			p.From3.Type = obj.TYPE_CONST
			p.From3.Offset = cnt
		}

		switch dir {
		case _FORWARDS:
			offset += cnt
		case _BACKWARDS:
			offset -= cnt
		}
		w -= cnt
	}
}
