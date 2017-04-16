// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"crypto/sha256"
	"fmt"
	"io"
	"strconv"
)

// architecture-independent object file output
const (
	ArhdrSize = 60
)

func formathdr(arhdr []byte, name string, size int64) {
	copy(arhdr[:], fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", name, 0, 0, 0, 0644, size))
}

// These modes say which kind of object file to generate.
// The default use of the toolchain is to set both bits,
// generating a combined compiler+linker object, one that
// serves to describe the package to both the compiler and the linker.
// In fact the compiler and linker read nearly disjoint sections of
// that file, though, so in a distributed build setting it can be more
// efficient to split the output into two files, supplying the compiler
// object only to future compilations and the linker object only to
// future links.
//
// By default a combined object is written, but if -linkobj is specified
// on the command line then the default -o output is a compiler object
// and the -linkobj output is a linker object.
const (
	modeCompilerObj = 1 << iota
	modeLinkerObj
)

func dumpobj() {
	if !dolinkobj {
		dumpobj1(outfile, modeCompilerObj)
		return
	}
	if linkobj == "" {
		dumpobj1(outfile, modeCompilerObj|modeLinkerObj)
		return
	}
	dumpobj1(outfile, modeCompilerObj)
	dumpobj1(linkobj, modeLinkerObj)
}

func dumpobj1(outfile string, mode int) {
	var err error
	bout, err = bio.Create(outfile)
	if err != nil {
		flusherrors()
		fmt.Printf("can't create %s: %v\n", outfile, err)
		errorexit()
	}

	startobj := int64(0)
	var arhdr [ArhdrSize]byte
	if writearchive {
		bout.WriteString("!<arch>\n")
		arhdr = [ArhdrSize]byte{}
		bout.Write(arhdr[:])
		startobj = bout.Offset()
	}

	printheader := func() {
		fmt.Fprintf(bout, "go object %s %s %s %s\n", objabi.GOOS, objabi.GOARCH, objabi.Version, objabi.Expstring())
		if buildid != "" {
			fmt.Fprintf(bout, "build id %q\n", buildid)
		}
		if localpkg.Name == "main" {
			fmt.Fprintf(bout, "main\n")
		}
		if safemode {
			fmt.Fprintf(bout, "safe\n")
		} else {
			fmt.Fprintf(bout, "----\n") // room for some other tool to write "safe"
		}
		fmt.Fprintf(bout, "\n") // header ends with blank line
	}

	printheader()

	if mode&modeCompilerObj != 0 {
		dumpexport()
	}

	if writearchive {
		bout.Flush()
		size := bout.Offset() - startobj
		if size&1 != 0 {
			bout.WriteByte(0)
		}
		bout.Seek(startobj-ArhdrSize, 0)
		formathdr(arhdr[:], "__.PKGDEF", size)
		bout.Write(arhdr[:])
		bout.Flush()
		bout.Seek(startobj+size+(size&1), 0)
	}

	if mode&modeLinkerObj == 0 {
		bout.Close()
		return
	}

	if writearchive {
		// start object file
		arhdr = [ArhdrSize]byte{}
		bout.Write(arhdr[:])
		startobj = bout.Offset()
		printheader()
	}

	if pragcgobuf != "" {
		if writearchive {
			// write empty export section; must be before cgo section
			fmt.Fprintf(bout, "\n$$\n\n$$\n\n")
		}

		fmt.Fprintf(bout, "\n$$  // cgo\n")
		fmt.Fprintf(bout, "%s\n$$\n\n", pragcgobuf)
	}

	fmt.Fprintf(bout, "\n!\n")

	externs := len(externdcl)

	dumpglobls()
	addptabs()
	addsignats(externdcl)
	dumpsignats()
	dumptabs()
	dumpimportstrings()
	dumpbasictypes()

	// Dump extra globals.
	tmp := externdcl

	if externdcl != nil {
		externdcl = externdcl[externs:]
	}
	dumpglobls()
	externdcl = tmp

	if zerosize > 0 {
		zero := mappkg.Lookup("zero")
		ggloblsym(zero.Linksym(), int32(zerosize), obj.DUPOK|obj.RODATA)
	}

	addGCLocals()

	obj.WriteObjFile(Ctxt, bout.Writer)

	if writearchive {
		bout.Flush()
		size := bout.Offset() - startobj
		if size&1 != 0 {
			bout.WriteByte(0)
		}
		bout.Seek(startobj-ArhdrSize, 0)
		formathdr(arhdr[:], "_go_.o", size)
		bout.Write(arhdr[:])
	}

	bout.Close()
}

func addptabs() {
	if !Ctxt.Flag_dynlink || localpkg.Name != "main" {
		return
	}
	for _, exportn := range exportlist {
		s := exportn.Sym
		n := asNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op != ONAME {
			continue
		}
		if !exportname(s.Name) {
			continue
		}
		if s.Pkg.Name != "main" {
			continue
		}
		if n.Type.Etype == TFUNC && n.Class() == PFUNC {
			// function
			ptabs = append(ptabs, ptabEntry{s: s, t: asNode(s.Def).Type})
		} else {
			// variable
			ptabs = append(ptabs, ptabEntry{s: s, t: types.NewPtr(asNode(s.Def).Type)})
		}
	}
}

func dumpglobls() {
	// add globals
	for _, n := range externdcl {
		if n.Op != ONAME {
			continue
		}

		if n.Type == nil {
			Fatalf("external %v nil type\n", n)
		}
		if n.Class() == PFUNC {
			continue
		}
		if n.Sym.Pkg != localpkg {
			continue
		}
		dowidth(n.Type)
		ggloblnod(n)
	}

	obj.SortSlice(funcsyms, func(i, j int) bool {
		return funcsyms[i].LinksymName() < funcsyms[j].LinksymName()
	})
	for _, s := range funcsyms {
		sf := s.Pkg.Lookup(funcsymname(s)).Linksym()
		dsymptr(sf, 0, s.Linksym(), 0)
		ggloblsym(sf, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// Do not reprocess funcsyms on next dumpglobls call.
	funcsyms = nil
}

// addGCLocals adds gcargs and gclocals symbols to Ctxt.Data.
// It takes care not to add any duplicates.
// Though the object file format handles duplicates efficiently,
// storing only a single copy of the data,
// failure to remove these duplicates adds a few percent to object file size.
func addGCLocals() {
	seen := make(map[string]bool)
	for _, s := range Ctxt.Text {
		if s.Func == nil {
			continue
		}
		for _, gcsym := range []*obj.LSym{&s.Func.GCArgs, &s.Func.GCLocals} {
			if seen[gcsym.Name] {
				continue
			}
			Ctxt.Data = append(Ctxt.Data, gcsym)
			seen[gcsym.Name] = true
		}
	}
}

func duintxx(s *obj.LSym, off int, v uint64, wid int) int {
	if off&(wid-1) != 0 {
		Fatalf("duintxxLSym: misaligned: v=%d wid=%d off=%d", v, wid, off)
	}
	s.WriteInt(Ctxt, int64(off), wid, int64(v))
	return off + wid
}

func duint8(s *obj.LSym, off int, v uint8) int {
	return duintxx(s, off, uint64(v), 1)
}

func duint16(s *obj.LSym, off int, v uint16) int {
	return duintxx(s, off, uint64(v), 2)
}

func duint32(s *obj.LSym, off int, v uint32) int {
	return duintxx(s, off, uint64(v), 4)
}

func duintptr(s *obj.LSym, off int, v uint64) int {
	return duintxx(s, off, v, Widthptr)
}

func dbvec(s *obj.LSym, off int, bv bvec) int {
	// Runtime reads the bitmaps as byte arrays. Oblige.
	for j := 0; int32(j) < bv.n; j += 8 {
		word := bv.b[j/32]
		off = duint8(s, off, uint8(word>>(uint(j)%32)))
	}
	return off
}

func stringsym(s string) (data *obj.LSym) {
	var symname string
	if len(s) > 100 {
		// Huge strings are hashed to avoid long names in object files.
		// Indulge in some paranoia by writing the length of s, too,
		// as protection against length extension attacks.
		h := sha256.New()
		io.WriteString(h, s)
		symname = fmt.Sprintf(".gostring.%d.%x", len(s), h.Sum(nil))
	} else {
		// Small strings get named directly by their contents.
		symname = strconv.Quote(s)
	}

	const prefix = "go.string."
	symdataname := prefix + symname

	symdata := Ctxt.Lookup(symdataname)

	if !symdata.SeenGlobl() {
		// string data
		off := dsname(symdata, 0, s)
		ggloblsym(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)
	}

	return symdata
}

var slicebytes_gen int

func slicebytes(nam *Node, s string, len int) {
	slicebytes_gen++
	symname := fmt.Sprintf(".gobytes.%d", slicebytes_gen)
	sym := localpkg.Lookup(symname)
	sym.Def = asTypesNode(newname(sym))

	lsym := sym.Linksym()
	off := dsname(lsym, 0, s)
	ggloblsym(lsym, int32(off), obj.NOPTR|obj.LOCAL)

	if nam.Op != ONAME {
		Fatalf("slicebytes %v", nam)
	}
	nsym := nam.Sym.Linksym()
	off = int(nam.Xoffset)
	off = dsymptr(nsym, off, lsym, 0)
	off = duintptr(nsym, off, uint64(len))
	duintptr(nsym, off, uint64(len))
}

func dsname(s *obj.LSym, off int, t string) int {
	s.WriteString(Ctxt, int64(off), len(t), t)
	return off + len(t)
}

func dsymptr(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	off = int(Rnd(int64(off), int64(Widthptr)))
	s.WriteAddr(Ctxt, int64(off), Widthptr, x, int64(xoff))
	off += Widthptr
	return off
}

func dsymptrOff(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	s.WriteOff(Ctxt, int64(off), x, int64(xoff))
	off += 4
	return off
}

func dsymptrWeakOff(s *obj.LSym, off int, x *obj.LSym) int {
	s.WriteWeakOff(Ctxt, int64(off), x, 0)
	off += 4
	return off
}

func gdata(nam *Node, nr *Node, wid int) {
	if nam.Op != ONAME {
		Fatalf("gdata nam op %v", nam.Op)
	}
	if nam.Sym == nil {
		Fatalf("gdata nil nam sym")
	}
	s := nam.Sym.Linksym()

	switch nr.Op {
	case OLITERAL:
		switch u := nr.Val().U.(type) {
		case bool:
			i := int64(obj.Bool2int(u))
			s.WriteInt(Ctxt, nam.Xoffset, wid, i)

		case *Mpint:
			s.WriteInt(Ctxt, nam.Xoffset, wid, u.Int64())

		case *Mpflt:
			f := u.Float64()
			switch nam.Type.Etype {
			case TFLOAT32:
				s.WriteFloat32(Ctxt, nam.Xoffset, float32(f))
			case TFLOAT64:
				s.WriteFloat64(Ctxt, nam.Xoffset, f)
			}

		case *Mpcplx:
			r := u.Real.Float64()
			i := u.Imag.Float64()
			switch nam.Type.Etype {
			case TCOMPLEX64:
				s.WriteFloat32(Ctxt, nam.Xoffset, float32(r))
				s.WriteFloat32(Ctxt, nam.Xoffset+4, float32(i))
			case TCOMPLEX128:
				s.WriteFloat64(Ctxt, nam.Xoffset, r)
				s.WriteFloat64(Ctxt, nam.Xoffset+8, i)
			}

		case string:
			symdata := stringsym(u)
			s.WriteAddr(Ctxt, nam.Xoffset, Widthptr, symdata, 0)
			s.WriteInt(Ctxt, nam.Xoffset+int64(Widthptr), Widthptr, int64(len(u)))

		default:
			Fatalf("gdata unhandled OLITERAL %v", nr)
		}

	case OADDR:
		if nr.Left.Op != ONAME {
			Fatalf("gdata ADDR left op %v", nr.Left.Op)
		}
		to := nr.Left
		s.WriteAddr(Ctxt, nam.Xoffset, wid, to.Sym.Linksym(), to.Xoffset)

	case ONAME:
		if nr.Class() != PFUNC {
			Fatalf("gdata NAME not PFUNC %d", nr.Class())
		}
		s.WriteAddr(Ctxt, nam.Xoffset, wid, funcsym(nr.Sym).Linksym(), nr.Xoffset)

	default:
		Fatalf("gdata unhandled op %v %v\n", nr, nr.Op)
	}
}
