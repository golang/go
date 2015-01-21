// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Objwriter reads an object file description in an unspecified format
// and writes a Go object file. It is invoked by parts of the toolchain
// that have not yet been converted from C to Go and should not be
// used otherwise.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"

	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"cmd/internal/obj/i386"
	"cmd/internal/obj/ppc64"
	"cmd/internal/obj/x86"
)

var arch *obj.LinkArch
var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to this file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

func main() {
	log.SetPrefix("goobj: ")
	log.SetFlags(0)
	flag.Parse()

	if flag.NArg() == 1 && flag.Arg(0) == "ping" {
		// old invocation from liblink, just testing that objwriter exists
		return
	}

	if flag.NArg() != 4 {
		fmt.Fprintf(os.Stderr, "usage: goobj infile objfile offset goarch\n")
		os.Exit(2)
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal(err)
		}
		defer pprof.WriteHeapProfile(f)
	}

	switch flag.Arg(3) {
	case "amd64":
		arch = &x86.Linkamd64
	case "amd64p32":
		arch = &x86.Linkamd64p32
	case "386":
		// TODO(rsc): Move Link386 to package x86.
		arch = &i386.Link386
	case "arm":
		arch = &arm.Linkarm
	case "ppc64":
		arch = &ppc64.Linkppc64
	case "ppc64le":
		arch = &ppc64.Linkppc64le
	}

	input()
}

const (
	// must match liblink/objfilego.c
	TypeEnd = iota
	TypeCtxt
	TypePlist
	TypeSym
	TypeProg
	TypeAddr
	TypeHist
)

var (
	ctxt   *obj.Link
	plists = map[int64]*obj.Plist{}
	syms   = map[int64]*obj.LSym{}
	progs  = map[int64]*obj.Prog{}
	hists  = map[int64]*obj.Hist{}
	undef  = map[interface{}]bool{}
)

func input() {
	args := flag.Args()
	ctxt = obj.Linknew(arch)
	ctxt.Debugasm = 1
	ctxt.Bso = obj.Binitw(os.Stdout)
	defer obj.Bflush(ctxt.Bso)
	ctxt.Diag = log.Fatalf
	f, err := os.Open(args[0])
	if err != nil {
		log.Fatal(err)
	}

	b := bufio.NewReaderSize(f, 1<<20)
	if v := rdint(b); v != TypeCtxt {
		log.Fatalf("invalid input - missing ctxt - got %d", v)
	}
	name := rdstring(b)
	if name != ctxt.Arch.Name {
		log.Fatalf("bad arch %s - want %s", name, ctxt.Arch.Name)
	}

	ctxt.Goarm = int32(rdint(b))
	ctxt.Debugasm = int32(rdint(b))
	ctxt.Trimpath = rdstring(b)
	ctxt.Plist = rdplist(b)
	ctxt.Plast = rdplist(b)
	ctxt.Hist = rdhist(b)
	ctxt.Ehist = rdhist(b)
	for {
		i := rdint(b)
		if i < 0 {
			break
		}
		ctxt.Hash[i] = rdsym(b)
	}
	last := int64(TypeCtxt)

Loop:
	for {
		t := rdint(b)
		switch t {
		default:
			log.Fatalf("unexpected input after type %d: %v", last, t)
		case TypeEnd:
			break Loop
		case TypePlist:
			readplist(b, rdplist(b))
		case TypeSym:
			readsym(b, rdsym(b))
		case TypeProg:
			readprog(b, rdprog(b))
		case TypeHist:
			readhist(b, rdhist(b))
		}
		last = t
	}

	if len(undef) > 0 {
		panic("missing definitions")
	}

	var buf bytes.Buffer
	obuf := obj.Binitw(&buf)
	obj.Writeobjdirect(ctxt, obuf)
	obj.Bflush(obuf)

	data, err := ioutil.ReadFile(args[1])
	if err != nil {
		log.Fatal(err)
	}

	offset, err := strconv.Atoi(args[2])
	if err != nil {
		log.Fatalf("bad offset: %v", err)
	}
	if offset > len(data) {
		log.Fatalf("offset too large: %v > %v", offset, len(data))
	}

	old := data[offset:]
	if len(old) > 0 && !bytes.Equal(old, buf.Bytes()) {
		out := strings.TrimSuffix(args[0], ".in") + ".out"
		if err := ioutil.WriteFile(out, append(data[:offset:offset], buf.Bytes()...), 0666); err != nil {
			log.Fatal(err)
		}
		log.Fatalf("goobj produced different output:\n\toriginal: %s\n\tgoobj: %s", args[1], out)
	}

	if len(old) == 0 {
		data = append(data, buf.Bytes()...)
		if err := ioutil.WriteFile(args[1], data, 0666); err != nil {
			log.Fatal(err)
		}
	}
}

func rdstring(b *bufio.Reader) string {
	v := rdint(b)
	buf := make([]byte, v)
	io.ReadFull(b, buf)
	return string(buf)
}

func rdint(b *bufio.Reader) int64 {
	var v uint64
	shift := uint(0)
	for {
		b, err := b.ReadByte()
		if err != nil {
			log.Fatal(err)
		}
		v |= uint64(b&0x7F) << shift
		shift += 7
		if b&0x80 == 0 {
			break
		}
	}
	return int64(v>>1) ^ int64(v<<63)>>63
}

func rdplist(b *bufio.Reader) *obj.Plist {
	id := rdint(b)
	if id == 0 {
		return nil
	}
	pl := plists[id]
	if pl == nil {
		pl = new(obj.Plist)
		plists[id] = pl
		undef[pl] = true
	}
	return pl
}

func rdsym(b *bufio.Reader) *obj.LSym {
	id := rdint(b)
	if id == 0 {
		return nil
	}
	sym := syms[id]
	if sym == nil {
		sym = new(obj.LSym)
		syms[id] = sym
		undef[sym] = true
	}
	return sym
}

func rdprog(b *bufio.Reader) *obj.Prog {
	id := rdint(b)
	if id == 0 {
		return nil
	}
	prog := progs[id]
	if prog == nil {
		prog = new(obj.Prog)
		prog.Ctxt = ctxt
		progs[id] = prog
		undef[prog] = true
	}
	return prog
}

func rdhist(b *bufio.Reader) *obj.Hist {
	id := rdint(b)
	if id == 0 {
		return nil
	}
	h := hists[id]
	if h == nil {
		h = new(obj.Hist)
		hists[id] = h
		undef[h] = true
	}
	return h
}

func readplist(b *bufio.Reader, pl *obj.Plist) {
	if !undef[pl] {
		panic("double-def")
	}
	delete(undef, pl)
	pl.Recur = int(rdint(b))
	pl.Name = rdsym(b)
	pl.Firstpc = rdprog(b)
	pl.Link = rdplist(b)
}

func readsym(b *bufio.Reader, s *obj.LSym) {
	if !undef[s] {
		panic("double-def")
	}
	delete(undef, s)
	s.Name = rdstring(b)
	s.Extname = rdstring(b)
	s.Type = int16(rdint(b))
	s.Version = int16(rdint(b))
	s.Dupok = uint8(rdint(b))
	s.External = uint8(rdint(b))
	s.Nosplit = uint8(rdint(b))
	s.Reachable = uint8(rdint(b))
	s.Cgoexport = uint8(rdint(b))
	s.Special = uint8(rdint(b))
	s.Stkcheck = uint8(rdint(b))
	s.Hide = uint8(rdint(b))
	s.Leaf = uint8(rdint(b))
	s.Fnptr = uint8(rdint(b))
	s.Seenglobl = uint8(rdint(b))
	s.Onlist = uint8(rdint(b))
	s.Symid = int16(rdint(b))
	s.Dynid = int32(rdint(b))
	s.Sig = int32(rdint(b))
	s.Plt = int32(rdint(b))
	s.Got = int32(rdint(b))
	s.Align = int32(rdint(b))
	s.Elfsym = int32(rdint(b))
	s.Args = int32(rdint(b))
	s.Locals = int32(rdint(b))
	s.Value = rdint(b)
	s.Size = rdint(b)
	s.Hash = rdsym(b)
	s.Allsym = rdsym(b)
	s.Next = rdsym(b)
	s.Sub = rdsym(b)
	s.Outer = rdsym(b)
	s.Gotype = rdsym(b)
	s.Reachparent = rdsym(b)
	s.Queue = rdsym(b)
	s.File = rdstring(b)
	s.Dynimplib = rdstring(b)
	s.Dynimpvers = rdstring(b)
	s.Text = rdprog(b)
	s.Etext = rdprog(b)
	n := int(rdint(b))
	if n > 0 {
		s.P = make([]byte, n)
		io.ReadFull(b, s.P)
	}
	s.R = make([]obj.Reloc, int(rdint(b)))
	for i := range s.R {
		r := &s.R[i]
		r.Off = int32(rdint(b))
		r.Siz = uint8(rdint(b))
		r.Done = uint8(rdint(b))
		r.Type = int32(rdint(b))
		r.Add = rdint(b)
		r.Xadd = rdint(b)
		r.Sym = rdsym(b)
		r.Xsym = rdsym(b)
	}
}

func readprog(b *bufio.Reader, p *obj.Prog) {
	if !undef[p] {
		panic("double-def")
	}
	delete(undef, p)
	p.Pc = rdint(b)
	p.Lineno = int32(rdint(b))
	p.Link = rdprog(b)
	p.As = int16(rdint(b))
	p.Reg = uint8(rdint(b))
	p.Scond = uint8(rdint(b))
	p.Width = int8(rdint(b))
	readaddr(b, &p.From)
	readaddr(b, &p.From3)
	readaddr(b, &p.To)
}

func readaddr(b *bufio.Reader, a *obj.Addr) {
	if rdint(b) != TypeAddr {
		log.Fatal("out of sync")
	}
	a.Offset = rdint(b)
	a.U.Dval = rdfloat(b)
	buf := make([]byte, 8)
	io.ReadFull(b, buf)
	a.U.Sval = string(buf)
	a.U.Branch = rdprog(b)
	a.Sym = rdsym(b)
	a.Gotype = rdsym(b)
	a.Type = int16(rdint(b))
	a.Index = uint8(rdint(b))
	a.Scale = int8(rdint(b))
	a.Reg = int8(rdint(b))
	a.Name = int8(rdint(b))
	a.Class = int8(rdint(b))
	a.Etype = uint8(rdint(b))
	a.Offset2 = int32(rdint(b))
	a.Width = rdint(b)
}

func readhist(b *bufio.Reader, h *obj.Hist) {
	if !undef[h] {
		panic("double-def")
	}
	delete(undef, h)
	h.Link = rdhist(b)
	h.Name = rdstring(b)
	h.Line = int32(rdint(b))
	h.Offset = int32(rdint(b))
}

func rdfloat(b *bufio.Reader) float64 {
	return math.Float64frombits(uint64(rdint(b)))
}
