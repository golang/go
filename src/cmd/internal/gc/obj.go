// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

/*
 * architecture-independent object file output
 */
const (
	ArhdrSize = 60
)

func formathdr(arhdr []byte, name string, size int64) {
	copy(arhdr[:], fmt.Sprintf("%-16s%-12d%-6d%-6d%-8o%-10d`\n", name, 0, 0, 0, 0644, size))
}

func dumpobj() {
	var err error
	bout, err = obj.Bopenw(outfile)
	if err != nil {
		Flusherrors()
		fmt.Printf("can't create %s: %v\n", outfile, err)
		errorexit()
	}

	startobj := int64(0)
	var arhdr [ArhdrSize]byte
	if writearchive != 0 {
		obj.Bwritestring(bout, "!<arch>\n")
		arhdr = [ArhdrSize]byte{}
		obj.Bwrite(bout, arhdr[:])
		startobj = obj.Boffset(bout)
	}

	fmt.Fprintf(bout, "go object %s %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion(), obj.Expstring())
	dumpexport()

	if writearchive != 0 {
		obj.Bflush(bout)
		size := obj.Boffset(bout) - startobj
		if size&1 != 0 {
			obj.Bputc(bout, 0)
		}
		obj.Bseek(bout, startobj-ArhdrSize, 0)
		formathdr(arhdr[:], "__.PKGDEF", size)
		obj.Bwrite(bout, arhdr[:])
		obj.Bflush(bout)

		obj.Bseek(bout, startobj+size+(size&1), 0)
		arhdr = [ArhdrSize]byte{}
		obj.Bwrite(bout, arhdr[:])
		startobj = obj.Boffset(bout)
		fmt.Fprintf(bout, "go object %s %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion(), obj.Expstring())
	}

	if pragcgobuf != "" {
		if writearchive != 0 {
			// write empty export section; must be before cgo section
			fmt.Fprintf(bout, "\n$$\n\n$$\n\n")
		}

		fmt.Fprintf(bout, "\n$$  // cgo\n")
		fmt.Fprintf(bout, "%s\n$$\n\n", pragcgobuf)
	}

	fmt.Fprintf(bout, "\n!\n")

	externs := (*NodeList)(nil)
	if externdcl != nil {
		externs = externdcl.End
	}

	dumpglobls()
	dumptypestructs()

	// Dump extra globals.
	tmp := externdcl

	if externs != nil {
		externdcl = externs.Next
	}
	dumpglobls()
	externdcl = tmp

	zero := Pkglookup("zerovalue", Runtimepkg)
	ggloblsym(zero, int32(zerosize), obj.DUPOK|obj.RODATA)

	dumpdata()
	obj.Writeobjdirect(Ctxt, bout)

	if writearchive != 0 {
		obj.Bflush(bout)
		size := obj.Boffset(bout) - startobj
		if size&1 != 0 {
			obj.Bputc(bout, 0)
		}
		obj.Bseek(bout, startobj-ArhdrSize, 0)
		namebuf = fmt.Sprintf("_go_.%c", Thearch.Thechar)
		formathdr(arhdr[:], namebuf, size)
		obj.Bwrite(bout, arhdr[:])
	}

	obj.Bterm(bout)
}

func dumpglobls() {
	var n *Node

	// add globals
	for l := externdcl; l != nil; l = l.Next {
		n = l.N
		if n.Op != ONAME {
			continue
		}

		if n.Type == nil {
			Fatal("external %v nil type\n", Nconv(n, 0))
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

	for l := funcsyms; l != nil; l = l.Next {
		n = l.N
		dsymptr(n.Sym, 0, n.Sym.Def.Shortname.Sym, 0)
		ggloblsym(n.Sym, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// Do not reprocess funcsyms on next dumpglobls call.
	funcsyms = nil
}

func Bputname(b *obj.Biobuf, s *obj.LSym) {
	obj.Bwritestring(b, s.Name)
	obj.Bputc(b, 0)
}

func Linksym(s *Sym) *obj.LSym {
	if s == nil {
		return nil
	}
	if s.Lsym != nil {
		return s.Lsym
	}
	if isblanksym(s) {
		s.Lsym = obj.Linklookup(Ctxt, "_", 0)
	} else if s.Linkname != "" {
		s.Lsym = obj.Linklookup(Ctxt, s.Linkname, 0)
	} else {
		p := fmt.Sprintf("%s.%s", s.Pkg.Prefix, s.Name)
		s.Lsym = obj.Linklookup(Ctxt, p, 0)
	}

	return s.Lsym
}

func duintxx(s *Sym, off int, v uint64, wid int) int {
	// Update symbol data directly instead of generating a
	// DATA instruction that liblink will have to interpret later.
	// This reduces compilation time and memory usage.
	off = int(Rnd(int64(off), int64(wid)))

	return int(obj.Setuintxx(Ctxt, Linksym(s), int64(off), v, int64(wid)))
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

func duint64(s *Sym, off int, v uint64) int {
	return duintxx(s, off, v, 8)
}

func duintptr(s *Sym, off int, v uint64) int {
	return duintxx(s, off, v, Widthptr)
}

var stringsym_gen int

func stringsym(s string) *Sym {
	var tmp struct {
		lit Strlit
		buf string
	}
	var pkg *Pkg

	if len(s) > 100 {
		// huge strings are made static to avoid long names
		stringsym_gen++
		namebuf = fmt.Sprintf(".gostring.%d", stringsym_gen)

		pkg = localpkg
	} else {
		// small strings get named by their contents,
		// so that multiple modules using the same string
		// can share it.
		tmp.lit.S = s
		namebuf = fmt.Sprintf("\"%v\"", Zconv(&tmp.lit, 0))
		pkg = gostringpkg
	}

	sym := Pkglookup(namebuf, pkg)

	// SymUniq flag indicates that data is generated already
	if sym.Flags&SymUniq != 0 {
		return sym
	}
	sym.Flags |= SymUniq
	sym.Def = newname(sym)

	off := 0

	// string header
	off = dsymptr(sym, off, sym, Widthptr+Widthint)
	off = duintxx(sym, off, uint64(len(s)), Widthint)

	// string data
	var m int
	for n := 0; n < len(s); n += m {
		m = 8
		if m > len(s)-n {
			m = len(s) - n
		}
		off = dsname(sym, off, s[n:n+m])
	}

	off = duint8(sym, off, 0)                    // terminating NUL for runtime
	off = (off + Widthptr - 1) &^ (Widthptr - 1) // round to pointer alignment
	ggloblsym(sym, int32(off), obj.DUPOK|obj.RODATA)

	return sym
}

var slicebytes_gen int

func slicebytes(nam *Node, s string, len int) {
	var m int

	slicebytes_gen++
	namebuf = fmt.Sprintf(".gobytes.%d", slicebytes_gen)
	sym := Pkglookup(namebuf, localpkg)
	sym.Def = newname(sym)

	off := 0
	for n := 0; n < len; n += m {
		m = 8
		if m > len-n {
			m = len - n
		}
		off = dsname(sym, off, s[n:n+m])
	}

	ggloblsym(sym, int32(off), obj.NOPTR)

	if nam.Op != ONAME {
		Fatal("slicebytes %v", Nconv(nam, 0))
	}
	off = int(nam.Xoffset)
	off = dsymptr(nam.Sym, off, sym, 0)
	off = duintxx(nam.Sym, off, uint64(len), Widthint)
	duintxx(nam.Sym, off, uint64(len), Widthint)
}

func dstringptr(s *Sym, off int, str string) int {
	off = int(Rnd(int64(off), int64(Widthptr)))
	p := Thearch.Gins(obj.ADATA, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
	p.From.Offset = int64(off)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(Widthptr)

	Datastring(str+"\x00", &p.To) // TODO(rsc): Remove NUL
	p.To.Type = obj.TYPE_ADDR
	p.To.Etype = Simtype[TINT]
	off += Widthptr

	return off
}

/*
 * gobj.c
 */
func Datastring(s string, a *obj.Addr) {
	sym := stringsym(s)
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_EXTERN
	a.Sym = Linksym(sym)
	a.Node = sym.Def
	a.Offset = int64(Widthptr) + int64(Widthint) // skip header
	a.Etype = Simtype[TINT]
}

func datagostring(sval *Strlit, a *obj.Addr) {
	sym := stringsym(sval.S)
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_EXTERN
	a.Sym = Linksym(sym)
	a.Node = sym.Def
	a.Offset = 0 // header
	a.Etype = TSTRING
}

func dgostringptr(s *Sym, off int, str string) int {
	if str == "" {
		return duintptr(s, off, 0)
	}

	n := len(str)
	lit := new(Strlit)
	lit.S = str
	lit.S = lit.S[:n]
	return dgostrlitptr(s, off, lit)
}

func dgostrlitptr(s *Sym, off int, lit *Strlit) int {
	if lit == nil {
		return duintptr(s, off, 0)
	}

	off = int(Rnd(int64(off), int64(Widthptr)))
	p := Thearch.Gins(obj.ADATA, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
	p.From.Offset = int64(off)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(Widthptr)
	datagostring(lit, &p.To)
	p.To.Type = obj.TYPE_ADDR
	p.To.Etype = Simtype[TINT]
	off += Widthptr

	return off
}

func dsname(s *Sym, off int, t string) int {
	p := Thearch.Gins(obj.ADATA, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Offset = int64(off)
	p.From.Sym = Linksym(s)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(len(t))

	p.To.Type = obj.TYPE_SCONST
	p.To.U.Sval = t
	return off + len(t)
}

func dsymptr(s *Sym, off int, x *Sym, xoff int) int {
	off = int(Rnd(int64(off), int64(Widthptr)))

	p := Thearch.Gins(obj.ADATA, nil, nil)
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_EXTERN
	p.From.Sym = Linksym(s)
	p.From.Offset = int64(off)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(Widthptr)
	p.To.Type = obj.TYPE_ADDR
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = Linksym(x)
	p.To.Offset = int64(xoff)
	off += Widthptr

	return off
}

func gdata(nam *Node, nr *Node, wid int) {
	if nr.Op == OLITERAL {
		switch nr.Val.Ctype {
		case CTCPLX:
			gdatacomplex(nam, nr.Val.U.Cval)
			return

		case CTSTR:
			gdatastring(nam, nr.Val.U.Sval)
			return
		}
	}

	p := Thearch.Gins(obj.ADATA, nam, nr)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(wid)
}

func gdatacomplex(nam *Node, cval *Mpcplx) {
	w := cplxsubtype(int(nam.Type.Etype))
	w = int(Types[w].Width)

	p := Thearch.Gins(obj.ADATA, nam, nil)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(w)
	p.To.Type = obj.TYPE_FCONST
	p.To.U.Dval = mpgetflt(&cval.Real)

	p = Thearch.Gins(obj.ADATA, nam, nil)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(w)
	p.From.Offset += int64(w)
	p.To.Type = obj.TYPE_FCONST
	p.To.U.Dval = mpgetflt(&cval.Imag)
}

func gdatastring(nam *Node, sval *Strlit) {
	var nod1 Node

	p := Thearch.Gins(obj.ADATA, nam, nil)
	Datastring(sval.S, &p.To)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = Types[Tptr].Width
	p.To.Type = obj.TYPE_ADDR

	//print("%P\n", p);

	Nodconst(&nod1, Types[TINT], int64(len(sval.S)))

	p = Thearch.Gins(obj.ADATA, nam, &nod1)
	p.From3.Type = obj.TYPE_CONST
	p.From3.Offset = int64(Widthint)
	p.From.Offset += int64(Widthptr)
}
