// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

// Reading of Go object files.
//
// Originally, Go object files were Plan 9 object files, but no longer.
// Now they are more like standard object files, in that each symbol is defined
// by an associated memory image (bytes) and a list of relocations to apply
// during linking. We do not (yet?) use a standard file format, however.
// For now, the format is chosen to be as simple as possible to read and write.
// It may change for reasons of efficiency, or we may even switch to a
// standard file format if there are compelling benefits to doing so.
// See golang.org/s/go13linker for more background.
//
// The file format is:
//
//	- magic header: "\x00\x00go13ld"
//	- byte 1 - version number
//	- sequence of strings giving dependencies (imported packages)
//	- empty string (marks end of sequence)
//	- sequence of sybol references used by the defined symbols
//	- byte 0xff (marks end of sequence)
//	- sequence of defined symbols
//	- byte 0xff (marks end of sequence)
//	- magic footer: "\xff\xffgo13ld"
//
// All integers are stored in a zigzag varint format.
// See golang.org/s/go12symtab for a definition.
//
// Data blocks and strings are both stored as an integer
// followed by that many bytes.
//
// A symbol reference is a string name followed by a version.
//
// A symbol points to other symbols using an index into the symbol
// reference sequence. Index 0 corresponds to a nil LSym* pointer.
// In the symbol layout described below "symref index" stands for this
// index.
//
// Each symbol is laid out as the following fields (taken from LSym*):
//
//	- byte 0xfe (sanity check for synchronization)
//	- type [int]
//	- name & version [symref index]
//	- flags [int]
//		1 dupok
//	- size [int]
//	- gotype [symref index]
//	- p [data block]
//	- nr [int]
//	- r [nr relocations, sorted by off]
//
// If type == STEXT, there are a few more fields:
//
//	- args [int]
//	- locals [int]
//	- nosplit [int]
//	- flags [int]
//		1<<0 leaf
//		1<<1 C function
//		1<<2 function may call reflect.Type.Method
//	- nlocal [int]
//	- local [nlocal automatics]
//	- pcln [pcln table]
//
// Each relocation has the encoding:
//
//	- off [int]
//	- siz [int]
//	- type [int]
//	- add [int]
//	- xadd [int]
//	- sym [symref index]
//	- xsym [symref index]
//
// Each local has the encoding:
//
//	- asym [symref index]
//	- offset [int]
//	- type [int]
//	- gotype [symref index]
//
// The pcln table has the encoding:
//
//	- pcsp [data block]
//	- pcfile [data block]
//	- pcline [data block]
//	- npcdata [int]
//	- pcdata [npcdata data blocks]
//	- nfuncdata [int]
//	- funcdata [nfuncdata symref index]
//	- funcdatasym [nfuncdata ints]
//	- nfile [int]
//	- file [nfile symref index]
//
// The file layout and meaning of type integers are architecture-independent.
//
// TODO(rsc): The file format is good for a first pass but needs work.
//	- There are SymID in the object file that should really just be strings.
//	- The actual symbol memory images are interlaced with the symbol
//	  metadata. They should be separated, to reduce the I/O required to
//	  load just the metadata.

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"log"
	"strconv"
	"strings"
)

const (
	startmagic = "\x00\x00go13ld"
	endmagic   = "\xff\xffgo13ld"
)

func ldobjfile(ctxt *Link, f *obj.Biobuf, pkg string, length int64, pn string) {
	start := obj.Boffset(f)
	ctxt.Version++
	var buf [8]uint8
	obj.Bread(f, buf[:])
	if string(buf[:]) != startmagic {
		log.Fatalf("%s: invalid file start %x %x %x %x %x %x %x %x", pn, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7])
	}
	c := obj.Bgetc(f)
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

	ctxt.CurRefs = []*LSym{nil} // zeroth ref is nil
	for {
		c, err := f.Peek(1)
		if err != nil {
			log.Fatalf("%s: peeking: %v", pn, err)
		}
		if c[0] == 0xff {
			obj.Bgetc(f)
			break
		}
		readref(ctxt, f, pkg, pn)
	}

	for {
		c, err := f.Peek(1)
		if err != nil {
			log.Fatalf("%s: peeking: %v", pn, err)
		}
		if c[0] == 0xff {
			break
		}
		readsym(ctxt, f, pkg, pn)
	}

	buf = [8]uint8{}
	obj.Bread(f, buf[:])
	if string(buf[:]) != endmagic {
		log.Fatalf("%s: invalid file end", pn)
	}

	if obj.Boffset(f) != start+length {
		log.Fatalf("%s: unexpected end at %d, want %d", pn, int64(obj.Boffset(f)), int64(start+length))
	}
}

func readsym(ctxt *Link, f *obj.Biobuf, pkg string, pn string) {
	if obj.Bgetc(f) != 0xfe {
		log.Fatalf("readsym out of sync")
	}
	t := rdint(f)
	s := rdsym(ctxt, f, pkg)
	flags := rdint(f)
	dupok := flags&1 != 0
	local := flags&2 != 0
	size := rdint(f)
	typ := rdsym(ctxt, f, pkg)
	data := rddata(f)
	nreloc := rdint(f)

	var dup *LSym
	if s.Type != 0 && s.Type != obj.SXREF {
		if (t == obj.SDATA || t == obj.SBSS || t == obj.SNOPTRBSS) && len(data) == 0 && nreloc == 0 {
			if s.Size < int64(size) {
				s.Size = int64(size)
			}
			if typ != nil && s.Gotype == nil {
				s.Gotype = typ
			}
			return
		}

		if (s.Type == obj.SDATA || s.Type == obj.SBSS || s.Type == obj.SNOPTRBSS) && len(s.P) == 0 && len(s.R) == 0 {
			goto overwrite
		}
		if s.Type != obj.SBSS && s.Type != obj.SNOPTRBSS && !dupok && !s.Attr.DuplicateOK() {
			log.Fatalf("duplicate symbol %s (types %d and %d) in %s and %s", s.Name, s.Type, t, s.File, pn)
		}
		if len(s.P) > 0 {
			dup = s
			s = linknewsym(ctxt, ".dup", -1)
		}
	}

overwrite:
	s.File = pkg
	if dupok {
		s.Attr |= AttrDuplicateOK
	}
	if t == obj.SXREF {
		log.Fatalf("bad sxref")
	}
	if t == 0 {
		log.Fatalf("missing type for %s in %s", s.Name, pn)
	}
	if t == obj.SBSS && (s.Type == obj.SRODATA || s.Type == obj.SNOPTRBSS) {
		t = int(s.Type)
	}
	s.Type = int16(t)
	if s.Size < int64(size) {
		s.Size = int64(size)
	}
	s.Attr.Set(AttrLocal, local)
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
			r.Off = rdint32(f)
			r.Siz = rduint8(f)
			r.Type = rdint32(f)
			r.Add = rdint64(f)
			rdint64(f) // Xadd, ignored
			r.Sym = rdsym(ctxt, f, pkg)
			rdsym(ctxt, f, pkg) // Xsym, ignored
		}
	}

	if len(s.P) > 0 && dup != nil && len(dup.P) > 0 && strings.HasPrefix(s.Name, "gclocals·") {
		// content-addressed garbage collection liveness bitmap symbol.
		// double check for hash collisions.
		if !bytes.Equal(s.P, dup.P) {
			log.Fatalf("dupok hash collision for %s in %s and %s", s.Name, s.File, pn)
		}
	}

	if s.Type == obj.STEXT {
		s.Args = rdint32(f)
		s.Locals = rdint32(f)
		if rduint8(f) != 0 {
			s.Attr |= AttrNoSplit
		}
		flags := rdint(f)
		if flags&(1<<2) != 0 {
			s.Attr |= AttrReflectMethod
		}
		n := rdint(f)
		s.Autom = make([]Auto, n)
		for i := 0; i < n; i++ {
			s.Autom[i] = Auto{
				Asym:    rdsym(ctxt, f, pkg),
				Aoffset: rdint32(f),
				Name:    rdint16(f),
				Gotype:  rdsym(ctxt, f, pkg),
			}
		}

		s.Pcln = new(Pcln)
		pc := s.Pcln
		pc.Pcsp.P = rddata(f)
		pc.Pcfile.P = rddata(f)
		pc.Pcline.P = rddata(f)
		n = rdint(f)
		pc.Pcdata = make([]Pcdata, n)
		pc.Npcdata = n
		for i := 0; i < n; i++ {
			pc.Pcdata[i].P = rddata(f)
		}
		n = rdint(f)
		pc.Funcdata = make([]*LSym, n)
		pc.Funcdataoff = make([]int64, n)
		pc.Nfuncdata = n
		for i := 0; i < n; i++ {
			pc.Funcdata[i] = rdsym(ctxt, f, pkg)
		}
		for i := 0; i < n; i++ {
			pc.Funcdataoff[i] = rdint64(f)
		}
		n = rdint(f)
		pc.File = make([]*LSym, n)
		pc.Nfile = n
		for i := 0; i < n; i++ {
			pc.File[i] = rdsym(ctxt, f, pkg)
		}

		if dup == nil {
			if s.Attr.OnList() {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= AttrOnList
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
		if s.Attr.DuplicateOK() {
			fmt.Fprintf(ctxt.Bso, "dupok ")
		}
		if s.Attr.NoSplit() {
			fmt.Fprintf(ctxt.Bso, "nosplit ")
		}
		fmt.Fprintf(ctxt.Bso, "size=%d value=%d", int64(s.Size), int64(s.Value))
		if s.Type == obj.STEXT {
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

func readref(ctxt *Link, f *obj.Biobuf, pkg string, pn string) {
	if obj.Bgetc(f) != 0xfe {
		log.Fatalf("readsym out of sync")
	}
	name := rdsymName(f, pkg)
	v := rdint(f)
	if v != 0 && v != 1 {
		log.Fatalf("invalid symbol version %d", v)
	}
	if v == 1 {
		v = ctxt.Version
	}
	lsym := Linklookup(ctxt, name, v)
	ctxt.CurRefs = append(ctxt.CurRefs, lsym)
}

func rdint64(f *obj.Biobuf) int64 {
	var c int

	uv := uint64(0)
	for shift := 0; ; shift += 7 {
		if shift >= 64 {
			log.Fatalf("corrupt input")
		}
		c = obj.Bgetc(f)
		uv |= uint64(c&0x7F) << uint(shift)
		if c&0x80 == 0 {
			break
		}
	}

	return int64(uv>>1) ^ (int64(uint64(uv)<<63) >> 63)
}

func rdint(f *obj.Biobuf) int {
	n := rdint64(f)
	if int64(int(n)) != n {
		log.Panicf("%v out of range for int", n)
	}
	return int(n)
}

func rdint32(f *obj.Biobuf) int32 {
	n := rdint64(f)
	if int64(int32(n)) != n {
		log.Panicf("%v out of range for int32", n)
	}
	return int32(n)
}

func rdint16(f *obj.Biobuf) int16 {
	n := rdint64(f)
	if int64(int16(n)) != n {
		log.Panicf("%v out of range for int16", n)
	}
	return int16(n)
}

func rduint8(f *obj.Biobuf) uint8 {
	n := rdint64(f)
	if int64(uint8(n)) != n {
		log.Panicf("%v out of range for uint8", n)
	}
	return uint8(n)
}

// rdBuf is used by rdstring and rdsymName as scratch for reading strings.
var rdBuf []byte
var emptyPkg = []byte(`"".`)

func rdstring(f *obj.Biobuf) string {
	n := rdint(f)
	if len(rdBuf) < n {
		rdBuf = make([]byte, n)
	}
	obj.Bread(f, rdBuf[:n])
	return string(rdBuf[:n])
}

const rddataBufMax = 1 << 14

var rddataBuf = make([]byte, rddataBufMax)

func rddata(f *obj.Biobuf) []byte {
	var p []byte
	n := rdint(f)
	if n > rddataBufMax {
		p = make([]byte, n)
	} else {
		if len(rddataBuf) < n {
			rddataBuf = make([]byte, rddataBufMax)
		}
		p = rddataBuf[:n:n]
		rddataBuf = rddataBuf[n:]
	}
	obj.Bread(f, p)
	return p
}

// rdsymName reads a symbol name, replacing all "". with pkg.
func rdsymName(f *obj.Biobuf, pkg string) string {
	n := rdint(f)
	if n == 0 {
		rdint64(f)
		return ""
	}

	if len(rdBuf) < n {
		rdBuf = make([]byte, n, 2*n)
	}
	origName := rdBuf[:n]
	obj.Bread(f, origName)
	adjName := rdBuf[n:n]
	for {
		i := bytes.Index(origName, emptyPkg)
		if i == -1 {
			adjName = append(adjName, origName...)
			break
		}
		adjName = append(adjName, origName[:i]...)
		adjName = append(adjName, pkg...)
		adjName = append(adjName, '.')
		origName = origName[i+len(emptyPkg):]
	}
	name := string(adjName)
	if len(adjName) > len(rdBuf) {
		rdBuf = adjName // save the larger buffer for reuse
	}
	return name
}

func rdsym(ctxt *Link, f *obj.Biobuf, pkg string) *LSym {
	i := rdint(f)
	if i == 0 {
		return nil
	}

	s := ctxt.CurRefs[i]
	if s == nil || s.Version != 0 {
		return s
	}

	if s.Name[0] == '$' && len(s.Name) > 5 && s.Type == 0 {
		x, err := strconv.ParseUint(s.Name[5:], 16, 64)
		if err != nil {
			log.Panicf("failed to parse $-symbol %s: %v", s.Name, err)
		}
		s.Type = obj.SRODATA
		s.Attr |= AttrLocal
		switch s.Name[:5] {
		case "$f32.":
			if uint64(uint32(x)) != x {
				log.Panicf("$-symbol %s too large: %d", s.Name, x)
			}
			Adduint32(ctxt, s, uint32(x))
		case "$f64.", "$i64.":
			Adduint64(ctxt, s, x)
		default:
			log.Panicf("unrecognized $-symbol: %s", s.Name)
		}
		s.Attr.Set(AttrReachable, false)
	}
	if strings.HasPrefix(s.Name, "runtime.gcbits.") {
		s.Attr |= AttrLocal
	}
	return s
}
