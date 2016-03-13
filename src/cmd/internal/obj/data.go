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

func Symgrow(ctxt *Link, s *LSym, lsiz int64) {
	siz := int(lsiz)
	if int64(siz) != lsiz {
		log.Fatalf("Symgrow size %d too long", lsiz)
	}
	if len(s.P) >= siz {
		return
	}
	// TODO(dfc) append cap-len at once, rather than
	// one byte at a time.
	for cap(s.P) < siz {
		s.P = append(s.P[:cap(s.P)], 0)
	}
	s.P = s.P[:siz]
}

// prepwrite prepares to write data of size siz into s at offset off.
func (s *LSym) prepwrite(ctxt *Link, off, siz int64) {
	if off < 0 || siz < 0 || off >= 1<<30 || siz >= 100 {
		log.Fatalf("prepwrite: bad off=%d siz=%d", off, siz)
	}
	if s.Type == SBSS || s.Type == STLSBSS {
		ctxt.Diag("cannot supply data for BSS var")
	}
	Symgrow(ctxt, s, off+siz)
}

// WriteFloat32 writes f into s at offset off.
func (s *LSym) WriteFloat32(ctxt *Link, off int64, f float32) {
	s.prepwrite(ctxt, off, 4)
	ctxt.Arch.ByteOrder.PutUint32(s.P[off:], math.Float32bits(f))
}

// WriteFloat64 writes f into s at offset off.
func (s *LSym) WriteFloat64(ctxt *Link, off int64, f float64) {
	s.prepwrite(ctxt, off, 8)
	ctxt.Arch.ByteOrder.PutUint64(s.P[off:], math.Float64bits(f))
}

// WriteInt writes an integer i of size siz into s at offset off.
func (s *LSym) WriteInt(ctxt *Link, off, siz int64, i int64) {
	s.prepwrite(ctxt, off, siz)
	switch siz {
	default:
		ctxt.Diag("WriteInt bad integer: %d", siz)
	case 1:
		s.P[off] = byte(i)
	case 2:
		ctxt.Arch.ByteOrder.PutUint16(s.P[off:], uint16(i))
	case 4:
		ctxt.Arch.ByteOrder.PutUint32(s.P[off:], uint32(i))
	case 8:
		ctxt.Arch.ByteOrder.PutUint64(s.P[off:], uint64(i))
	}
}

// WriteAddr writes an address of size siz into s at offset off.
// rsym and roff specify the relocation for the address.
func (s *LSym) WriteAddr(ctxt *Link, off, siz int64, rsym *LSym, roff int64) {
	s.prepwrite(ctxt, off, siz)
	r := Addrel(s)
	r.Off = int32(off)
	r.Siz = uint8(siz)
	r.Sym = rsym
	r.Type = R_ADDR
	r.Add = roff
}

// WriteString writes a string of size siz into s at offset off.
func (s *LSym) WriteString(ctxt *Link, off, siz int64, str string) {
	s.prepwrite(ctxt, off, siz)
	copy(s.P[off:off+siz], str)
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
