// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
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

package obj

import (
	"log"
	"math"
)

func mangle(file string) {
	log.Fatalf("%s: mangled input file", file)
}

func Symgrow(ctxt *Link, s *LSym, lsiz int64) {
	siz := int(lsiz)
	if int64(siz) != lsiz {
		log.Fatalf("Symgrow size %d too long", lsiz)
	}
	if len(s.P) >= siz {
		return
	}
	for cap(s.P) < siz {
		s.P = append(s.P[:cap(s.P)], 0)
	}
	s.P = s.P[:siz]
}

func savedata(ctxt *Link, s *LSym, p *Prog, pn string) {
	off := int32(p.From.Offset)
	siz := int32(p.From3.Offset)
	if off < 0 || siz < 0 || off >= 1<<30 || siz >= 100 {
		mangle(pn)
	}
	if ctxt.Enforce_data_order != 0 && off < int32(len(s.P)) {
		ctxt.Diag("data out of order (already have %d)\n%v", len(s.P), p)
	}
	Symgrow(ctxt, s, int64(off+siz))

	switch int(p.To.Type) {
	default:
		ctxt.Diag("bad data: %v", p)

	case TYPE_FCONST:
		switch siz {
		default:
			ctxt.Diag("unexpected %d-byte floating point constant", siz)

		case 4:
			flt := math.Float32bits(float32(p.To.Val.(float64)))
			ctxt.Arch.ByteOrder.PutUint32(s.P[off:], flt)

		case 8:
			flt := math.Float64bits(p.To.Val.(float64))
			ctxt.Arch.ByteOrder.PutUint64(s.P[off:], flt)
		}

	case TYPE_SCONST:
		copy(s.P[off:off+siz], p.To.Val.(string))

	case TYPE_CONST, TYPE_ADDR:
		if p.To.Sym != nil || int(p.To.Type) == TYPE_ADDR {
			r := Addrel(s)
			r.Off = off
			r.Siz = uint8(siz)
			r.Sym = p.To.Sym
			r.Type = R_ADDR
			r.Add = p.To.Offset
			break
		}
		o := p.To.Offset
		switch siz {
		default:
			ctxt.Diag("unexpected %d-byte integer constant", siz)
		case 1:
			s.P[off] = byte(o)
		case 2:
			ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(o))
		case 4:
			ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(o))
		case 8:
			ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(o))
		}
	}
}

func Addrel(s *LSym) *Reloc {
	s.R = append(s.R, Reloc{})
	return &s.R[len(s.R)-1]
}

func Setuintxx(ctxt *Link, s *LSym, off int64, v uint64, wid int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	if s.Size < off+wid {
		s.Size = off + wid
		Symgrow(ctxt, s, s.Size)
	}

	switch wid {
	case 1:
		s.P[off] = uint8(v)
	case 2:
		ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(v))
	case 4:
		ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(v))
	case 8:
		ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(v))
	}

	return off + wid
}

func adduintxx(ctxt *Link, s *LSym, v uint64, wid int) int64 {
	off := s.Size
	Setuintxx(ctxt, s, off, v, int64(wid))
	return off
}

func adduint8(ctxt *Link, s *LSym, v uint8) int64 {
	return adduintxx(ctxt, s, uint64(v), 1)
}

func adduint16(ctxt *Link, s *LSym, v uint16) int64 {
	return adduintxx(ctxt, s, uint64(v), 2)
}

func Adduint32(ctxt *Link, s *LSym, v uint32) int64 {
	return adduintxx(ctxt, s, uint64(v), 4)
}

func Adduint64(ctxt *Link, s *LSym, v uint64) int64 {
	return adduintxx(ctxt, s, v, 8)
}

func setuint8(ctxt *Link, s *LSym, r int64, v uint8) int64 {
	return Setuintxx(ctxt, s, r, uint64(v), 1)
}

func setuint16(ctxt *Link, s *LSym, r int64, v uint16) int64 {
	return Setuintxx(ctxt, s, r, uint64(v), 2)
}

func setuint32(ctxt *Link, s *LSym, r int64, v uint32) int64 {
	return Setuintxx(ctxt, s, r, uint64(v), 4)
}

func setuint64(ctxt *Link, s *LSym, r int64, v uint64) int64 {
	return Setuintxx(ctxt, s, r, v, 8)
}

func addaddrplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}

func addpcrelplus(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Add = add
	r.Type = R_PCREL
	r.Siz = 4
	return i + int64(r.Siz)
}

func addaddr(ctxt *Link, s *LSym, t *LSym) int64 {
	return addaddrplus(ctxt, s, t, 0)
}

func setaddrplus(ctxt *Link, s *LSym, off int64, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	if off+int64(ctxt.Arch.Ptrsize) > s.Size {
		s.Size = off + int64(ctxt.Arch.Ptrsize)
		Symgrow(ctxt, s, s.Size)
	}

	r := Addrel(s)
	r.Sym = t
	r.Off = int32(off)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_ADDR
	r.Add = add
	return off + int64(r.Siz)
}

func setaddr(ctxt *Link, s *LSym, off int64, t *LSym) int64 {
	return setaddrplus(ctxt, s, off, t, 0)
}

func addsize(ctxt *Link, s *LSym, t *LSym) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	i := s.Size
	s.Size += int64(ctxt.Arch.Ptrsize)
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = uint8(ctxt.Arch.Ptrsize)
	r.Type = R_SIZE
	return i + int64(r.Siz)
}

func addaddrplus4(ctxt *Link, s *LSym, t *LSym, add int64) int64 {
	if s.Type == 0 {
		s.Type = SDATA
	}
	i := s.Size
	s.Size += 4
	Symgrow(ctxt, s, s.Size)
	r := Addrel(s)
	r.Sym = t
	r.Off = int32(i)
	r.Siz = 4
	r.Type = R_ADDR
	r.Add = add
	return i + int64(r.Siz)
}
