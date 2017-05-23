// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/bio"
	"cmd/internal/obj"
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
	if linkobj == "" {
		dumpobj1(outfile, modeCompilerObj|modeLinkerObj)
	} else {
		dumpobj1(outfile, modeCompilerObj)
		dumpobj1(linkobj, modeLinkerObj)
	}
}

func dumpobj1(outfile string, mode int) {
	var err error
	bout, err = bio.Create(outfile)
	if err != nil {
		Flusherrors()
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
		fmt.Fprintf(bout, "go object %s %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion(), obj.Expstring())
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
	dumptypestructs()

	// Dump extra globals.
	tmp := externdcl

	if externdcl != nil {
		externdcl = externdcl[externs:]
	}
	dumpglobls()
	externdcl = tmp

	if zerosize > 0 {
		zero := Pkglookup("zero", mappkg)
		ggloblsym(zero, int32(zerosize), obj.DUPOK|obj.RODATA)
	}

	dumpdata()
	obj.Writeobjdirect(Ctxt, bout.Writer)

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

func dumpglobls() {
	// add globals
	for _, n := range externdcl {
		if n.Op != ONAME {
			continue
		}

		if n.Type == nil {
			Fatalf("external %v nil type\n", n)
		}
		if n.Class == PFUNC {
			continue
		}
		if n.Sym.Pkg != localpkg {
			continue
		}
		dowidth(n.Type)
		ggloblnod(n)
	}

	for _, n := range funcsyms {
		dsymptr(n.Sym, 0, n.Sym.Def.Func.Shortname.Sym, 0)
		ggloblsym(n.Sym, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// Do not reprocess funcsyms on next dumpglobls call.
	funcsyms = nil
}

func Linksym(s *Sym) *obj.LSym {
	if s == nil {
		return nil
	}
	if s.Lsym != nil {
		return s.Lsym
	}
	var name string
	if isblanksym(s) {
		name = "_"
	} else if s.Linkname != "" {
		name = s.Linkname
	} else {
		name = s.Pkg.Prefix + "." + s.Name
	}

	ls := obj.Linklookup(Ctxt, name, 0)
	s.Lsym = ls
	return ls
}

func duintxx(s *Sym, off int, v uint64, wid int) int {
	return duintxxLSym(Linksym(s), off, v, wid)
}

func duintxxLSym(s *obj.LSym, off int, v uint64, wid int) int {
	// Update symbol data directly instead of generating a
	// DATA instruction that liblink will have to interpret later.
	// This reduces compilation time and memory usage.
	off = int(Rnd(int64(off), int64(wid)))

	return int(obj.Setuintxx(Ctxt, s, int64(off), v, int64(wid)))
}

func duint8(s *Sym, off int, v uint8) int {
	return duintxx(s, off, uint64(v), 1)
}

func duint16(s *Sym, off int, v uint16) int {
	return duintxx(s, off, uint64(v), 2)
}

func duint32(s *Sym, off int, v uint32) int {
	return duintxx(s, off, uint64(v), 4)
}

func duintptr(s *Sym, off int, v uint64) int {
	return duintxx(s, off, v, Widthptr)
}

// stringConstantSyms holds the pair of symbols we create for a
// constant string.
type stringConstantSyms struct {
	hdr  *obj.LSym // string header
	data *obj.LSym // actual string data
}

// stringConstants maps from the symbol name we use for the string
// contents to the pair of linker symbols for that string.
var stringConstants = make(map[string]stringConstantSyms, 100)

func stringsym(s string) (hdr, data *obj.LSym) {
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

	// All the strings have the same prefix, so ignore it for map
	// purposes, but use a slice of the symbol name string to
	// reduce long-term memory overhead.
	key := symdataname[len(prefix):]

	if syms, ok := stringConstants[key]; ok {
		return syms.hdr, syms.data
	}

	symhdrname := "go.string.hdr." + symname

	symhdr := obj.Linklookup(Ctxt, symhdrname, 0)
	symdata := obj.Linklookup(Ctxt, symdataname, 0)

	stringConstants[key] = stringConstantSyms{symhdr, symdata}

	// string header
	off := 0
	off = dsymptrLSym(symhdr, off, symdata, 0)
	off = duintxxLSym(symhdr, off, uint64(len(s)), Widthint)
	ggloblLSym(symhdr, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)

	// string data
	off = dsnameLSym(symdata, 0, s)
	ggloblLSym(symdata, int32(off), obj.DUPOK|obj.RODATA|obj.LOCAL)

	return symhdr, symdata
}

var slicebytes_gen int

func slicebytes(nam *Node, s string, len int) {
	slicebytes_gen++
	symname := fmt.Sprintf(".gobytes.%d", slicebytes_gen)
	sym := Pkglookup(symname, localpkg)
	sym.Def = newname(sym)

	off := dsname(sym, 0, s)
	ggloblsym(sym, int32(off), obj.NOPTR|obj.LOCAL)

	if nam.Op != ONAME {
		Fatalf("slicebytes %v", nam)
	}
	off = int(nam.Xoffset)
	off = dsymptr(nam.Sym, off, sym, 0)
	off = duintxx(nam.Sym, off, uint64(len), Widthint)
	duintxx(nam.Sym, off, uint64(len), Widthint)
}

func Datastring(s string, a *obj.Addr) {
	_, symdata := stringsym(s)
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_EXTERN
	a.Sym = symdata
	a.Offset = 0
	a.Etype = uint8(Simtype[TINT])
}

func datagostring(sval string, a *obj.Addr) {
	symhdr, _ := stringsym(sval)
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_EXTERN
	a.Sym = symhdr
	a.Offset = 0
	a.Etype = uint8(TSTRING)
}

func dgostringptr(s *Sym, off int, str string) int {
	if str == "" {
		return duintptr(s, off, 0)
	}
	return dgostrlitptr(s, off, &str)
}

func dgostrlitptr(s *Sym, off int, lit *string) int {
	if lit == nil {
		return duintptr(s, off, 0)
	}
	off = int(Rnd(int64(off), int64(Widthptr)))
	symhdr, _ := stringsym(*lit)
	Linksym(s).WriteAddr(Ctxt, int64(off), Widthptr, symhdr, 0)
	off += Widthptr
	return off
}

func dsname(s *Sym, off int, t string) int {
	return dsnameLSym(Linksym(s), off, t)
}

func dsnameLSym(s *obj.LSym, off int, t string) int {
	s.WriteString(Ctxt, int64(off), len(t), t)
	return off + len(t)
}

func dsymptr(s *Sym, off int, x *Sym, xoff int) int {
	return dsymptrLSym(Linksym(s), off, Linksym(x), xoff)
}

func dsymptrLSym(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	off = int(Rnd(int64(off), int64(Widthptr)))
	s.WriteAddr(Ctxt, int64(off), Widthptr, x, int64(xoff))
	off += Widthptr
	return off
}

func dsymptrOffLSym(s *obj.LSym, off int, x *obj.LSym, xoff int) int {
	s.WriteOff(Ctxt, int64(off), x, int64(xoff))
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

	switch nr.Op {
	case OLITERAL:
		switch u := nr.Val().U.(type) {
		case *Mpcplx:
			gdatacomplex(nam, u)

		case string:
			gdatastring(nam, u)

		case bool:
			i := int64(obj.Bool2int(u))
			Linksym(nam.Sym).WriteInt(Ctxt, nam.Xoffset, wid, i)

		case *Mpint:
			Linksym(nam.Sym).WriteInt(Ctxt, nam.Xoffset, wid, u.Int64())

		case *Mpflt:
			s := Linksym(nam.Sym)
			f := u.Float64()
			switch nam.Type.Etype {
			case TFLOAT32:
				s.WriteFloat32(Ctxt, nam.Xoffset, float32(f))
			case TFLOAT64:
				s.WriteFloat64(Ctxt, nam.Xoffset, f)
			}

		default:
			Fatalf("gdata unhandled OLITERAL %v", nr)
		}

	case OADDR:
		if nr.Left.Op != ONAME {
			Fatalf("gdata ADDR left op %s", nr.Left.Op)
		}
		to := nr.Left
		Linksym(nam.Sym).WriteAddr(Ctxt, nam.Xoffset, wid, Linksym(to.Sym), to.Xoffset)

	case ONAME:
		if nr.Class != PFUNC {
			Fatalf("gdata NAME not PFUNC %d", nr.Class)
		}
		Linksym(nam.Sym).WriteAddr(Ctxt, nam.Xoffset, wid, Linksym(funcsym(nr.Sym)), nr.Xoffset)

	default:
		Fatalf("gdata unhandled op %v %v\n", nr, nr.Op)
	}
}

func gdatacomplex(nam *Node, cval *Mpcplx) {
	t := Types[cplxsubtype(nam.Type.Etype)]
	r := cval.Real.Float64()
	i := cval.Imag.Float64()
	s := Linksym(nam.Sym)

	switch t.Etype {
	case TFLOAT32:
		s.WriteFloat32(Ctxt, nam.Xoffset, float32(r))
		s.WriteFloat32(Ctxt, nam.Xoffset+4, float32(i))
	case TFLOAT64:
		s.WriteFloat64(Ctxt, nam.Xoffset, r)
		s.WriteFloat64(Ctxt, nam.Xoffset+8, i)
	}
}

func gdatastring(nam *Node, sval string) {
	s := Linksym(nam.Sym)
	_, symdata := stringsym(sval)
	s.WriteAddr(Ctxt, nam.Xoffset, Widthptr, symdata, 0)
	s.WriteInt(Ctxt, nam.Xoffset+int64(Widthptr), Widthint, int64(len(sval)))
}
