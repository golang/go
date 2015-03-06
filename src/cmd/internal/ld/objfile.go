// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"fmt"
	"log"
	"strconv"
	"strings"
)

const (
	startmagic = "\x00\x00go13ld"
	endmagic   = "\xff\xffgo13ld"
)

func ldobjfile(ctxt *Link, f *Biobuf, pkg string, length int64, pn string) {
	start := Boffset(f)
	ctxt.Version++
	var buf [8]uint8
	Bread(f, buf[:])
	if string(buf[:]) != startmagic {
		log.Fatalf("%s: invalid file start %x %x %x %x %x %x %x %x", pn, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7])
	}
	c := Bgetc(f)
	if c != 1 {
		log.Fatalf("%s: invalid file version number %d", pn, c)
	}

	var lib string
	for {
		lib = rdstring(f)
		if lib == "" {
			break
		}
		addlib(ctxt, pkg, pn, lib)
	}

	for {
		c = Bgetc(f)
		Bungetc(f)
		if c == 0xff {
			break
		}
		readsym(ctxt, f, pkg, pn)
	}

	buf = [8]uint8{}
	Bread(f, buf[:])
	if string(buf[:]) != endmagic {
		log.Fatalf("%s: invalid file end", pn)
	}

	if Boffset(f) != start+length {
		log.Fatalf("%s: unexpected end at %d, want %d", pn, int64(Boffset(f)), int64(start+length))
	}
}

var readsym_ndup int

func readsym(ctxt *Link, f *Biobuf, pkg string, pn string) {
	if Bgetc(f) != 0xfe {
		log.Fatalf("readsym out of sync")
	}
	t := int(rdint(f))
	name := expandpkg(rdstring(f), pkg)
	v := int(rdint(f))
	if v != 0 && v != 1 {
		log.Fatalf("invalid symbol version %d", v)
	}
	dupok := int(rdint(f))
	dupok &= 1
	size := int(rdint(f))
	typ := rdsym(ctxt, f, pkg)
	var data []byte
	rddata(f, &data)
	nreloc := int(rdint(f))

	if v != 0 {
		v = ctxt.Version
	}
	s := Linklookup(ctxt, name, v)
	var dup *LSym
	if s.Type != 0 && s.Type != SXREF {
		if (t == SDATA || t == SBSS || t == SNOPTRBSS) && len(data) == 0 && nreloc == 0 {
			if s.Size < int64(size) {
				s.Size = int64(size)
			}
			if typ != nil && s.Gotype == nil {
				s.Gotype = typ
			}
			return
		}

		if (s.Type == SDATA || s.Type == SBSS || s.Type == SNOPTRBSS) && len(s.P) == 0 && len(s.R) == 0 {
			goto overwrite
		}
		if s.Type != SBSS && s.Type != SNOPTRBSS && dupok == 0 && s.Dupok == 0 {
			log.Fatalf("duplicate symbol %s (types %d and %d) in %s and %s", s.Name, s.Type, t, s.File, pn)
		}
		if len(s.P) > 0 {
			dup = s
			s = linknewsym(ctxt, ".dup", readsym_ndup)
			readsym_ndup++ // scratch
		}
	}

overwrite:
	s.File = pkg
	s.Dupok = uint8(dupok)
	if t == SXREF {
		log.Fatalf("bad sxref")
	}
	if t == 0 {
		log.Fatalf("missing type for %s in %s", name, pn)
	}
	if t == SBSS && (s.Type == SRODATA || s.Type == SNOPTRBSS) {
		t = int(s.Type)
	}
	s.Type = int16(t)
	if s.Size < int64(size) {
		s.Size = int64(size)
	}
	if typ != nil { // if bss sym defined multiple times, take type from any one def
		s.Gotype = typ
	}
	if dup != nil && typ != nil {
		dup.Gotype = typ
	}
	s.P = data
	s.P = s.P[:len(data)]
	if nreloc > 0 {
		s.R = make([]Reloc, nreloc)
		s.R = s.R[:nreloc]
		var r *Reloc
		for i := 0; i < nreloc; i++ {
			r = &s.R[i]
			r.Off = int32(rdint(f))
			r.Siz = uint8(rdint(f))
			r.Type = int32(rdint(f))
			r.Add = rdint(f)
			r.Xadd = rdint(f)
			r.Sym = rdsym(ctxt, f, pkg)
			r.Xsym = rdsym(ctxt, f, pkg)
		}
	}

	if len(s.P) > 0 && dup != nil && len(dup.P) > 0 && strings.HasPrefix(s.Name, "gclocals·") {
		// content-addressed garbage collection liveness bitmap symbol.
		// double check for hash collisions.
		if !bytes.Equal(s.P, dup.P) {
			log.Fatalf("dupok hash collision for %s in %s and %s", s.Name, s.File, pn)
		}
	}

	if s.Type == STEXT {
		s.Args = int32(rdint(f))
		s.Locals = int32(rdint(f))
		s.Nosplit = uint8(rdint(f))
		v := int(rdint(f))
		s.Leaf = uint8(v & 1)
		s.Cfunc = uint8(v & 2)
		n := int(rdint(f))
		var a *Auto
		for i := 0; i < n; i++ {
			a = new(Auto)
			a.Asym = rdsym(ctxt, f, pkg)
			a.Aoffset = int32(rdint(f))
			a.Name = int16(rdint(f))
			a.Gotype = rdsym(ctxt, f, pkg)
			a.Link = s.Autom
			s.Autom = a
		}

		s.Pcln = new(Pcln)
		pc := s.Pcln
		rddata(f, &pc.Pcsp.P)
		rddata(f, &pc.Pcfile.P)
		rddata(f, &pc.Pcline.P)
		n = int(rdint(f))
		pc.Pcdata = make([]Pcdata, n)
		pc.Npcdata = n
		for i := 0; i < n; i++ {
			rddata(f, &pc.Pcdata[i].P)
		}
		n = int(rdint(f))
		pc.Funcdata = make([]*LSym, n)
		pc.Funcdataoff = make([]int64, n)
		pc.Nfuncdata = n
		for i := 0; i < n; i++ {
			pc.Funcdata[i] = rdsym(ctxt, f, pkg)
		}
		for i := 0; i < n; i++ {
			pc.Funcdataoff[i] = rdint(f)
		}
		n = int(rdint(f))
		pc.File = make([]*LSym, n)
		pc.Nfile = n
		for i := 0; i < n; i++ {
			pc.File[i] = rdsym(ctxt, f, pkg)
		}

		if dup == nil {
			if s.Onlist != 0 {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Onlist = 1
			if ctxt.Etextp != nil {
				ctxt.Etextp.Next = s
			} else {
				ctxt.Textp = s
			}
			ctxt.Etextp = s
		}
	}

	if ctxt.Debugasm != 0 {
		fmt.Fprintf(ctxt.Bso, "%s ", s.Name)
		if s.Version != 0 {
			fmt.Fprintf(ctxt.Bso, "v=%d ", s.Version)
		}
		if s.Type != 0 {
			fmt.Fprintf(ctxt.Bso, "t=%d ", s.Type)
		}
		if s.Dupok != 0 {
			fmt.Fprintf(ctxt.Bso, "dupok ")
		}
		if s.Cfunc != 0 {
			fmt.Fprintf(ctxt.Bso, "cfunc ")
		}
		if s.Nosplit != 0 {
			fmt.Fprintf(ctxt.Bso, "nosplit ")
		}
		fmt.Fprintf(ctxt.Bso, "size=%d value=%d", int64(s.Size), int64(s.Value))
		if s.Type == STEXT {
			fmt.Fprintf(ctxt.Bso, " args=%#x locals=%#x", uint64(s.Args), uint64(s.Locals))
		}
		fmt.Fprintf(ctxt.Bso, "\n")
		var c int
		var j int
		for i := 0; i < len(s.P); {
			fmt.Fprintf(ctxt.Bso, "\t%#04x", uint(i))
			for j = i; j < i+16 && j < len(s.P); j++ {
				fmt.Fprintf(ctxt.Bso, " %02x", s.P[j])
			}
			for ; j < i+16; j++ {
				fmt.Fprintf(ctxt.Bso, "   ")
			}
			fmt.Fprintf(ctxt.Bso, "  ")
			for j = i; j < i+16 && j < len(s.P); j++ {
				c = int(s.P[j])
				if ' ' <= c && c <= 0x7e {
					fmt.Fprintf(ctxt.Bso, "%c", c)
				} else {
					fmt.Fprintf(ctxt.Bso, ".")
				}
			}

			fmt.Fprintf(ctxt.Bso, "\n")
			i += 16
		}

		var r *Reloc
		for i := 0; i < len(s.R); i++ {
			r = &s.R[i]
			fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s+%d\n", int(r.Off), r.Siz, r.Type, r.Sym.Name, int64(r.Add))
		}
	}
}

func rdint(f *Biobuf) int64 {
	var c int

	uv := uint64(0)
	for shift := 0; ; shift += 7 {
		if shift >= 64 {
			log.Fatalf("corrupt input")
		}
		c = Bgetc(f)
		uv |= uint64(c&0x7F) << uint(shift)
		if c&0x80 == 0 {
			break
		}
	}

	return int64(uv>>1) ^ (int64(uint64(uv)<<63) >> 63)
}

func rdstring(f *Biobuf) string {
	n := rdint(f)
	p := make([]byte, n)
	Bread(f, p)
	return string(p)
}

func rddata(f *Biobuf, pp *[]byte) {
	n := rdint(f)
	*pp = make([]byte, n)
	Bread(f, *pp)
}

var symbuf []byte

func rdsym(ctxt *Link, f *Biobuf, pkg string) *LSym {
	n := int(rdint(f))
	if n == 0 {
		rdint(f)
		return nil
	}

	if len(symbuf) < n {
		symbuf = make([]byte, n)
	}
	Bread(f, symbuf[:n])
	p := string(symbuf[:n])
	v := int(rdint(f))
	if v != 0 {
		v = ctxt.Version
	}
	s := Linklookup(ctxt, expandpkg(p, pkg), v)

	if v == 0 && s.Name[0] == '$' && s.Type == 0 {
		if strings.HasPrefix(s.Name, "$f32.") {
			x, _ := strconv.ParseUint(s.Name[5:], 16, 32)
			i32 := int32(x)
			s.Type = SRODATA
			Adduint32(ctxt, s, uint32(i32))
			s.Reachable = false
		} else if strings.HasPrefix(s.Name, "$f64.") || strings.HasPrefix(s.Name, "$i64.") {
			x, _ := strconv.ParseUint(s.Name[5:], 16, 64)
			i64 := int64(x)
			s.Type = SRODATA
			Adduint64(ctxt, s, uint64(i64))
			s.Reachable = false
		}
	}

	return s
}
