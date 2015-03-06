// Inferno utils/8l/asm.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/asm.c
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

package ld

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
)

// Data layout and relocation.

// Derived from Inferno utils/6l/l.h
// http://code.google.com/p/inferno-os/source/browse/utils/6l/l.h
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

type Arch struct {
	Thechar          int
	Ptrsize          int
	Intsize          int
	Regsize          int
	Funcalign        int
	Maxalign         int
	Minlc            int
	Dwarfregsp       int
	Linuxdynld       string
	Freebsddynld     string
	Netbsddynld      string
	Openbsddynld     string
	Dragonflydynld   string
	Solarisdynld     string
	Adddynlib        func(string)
	Adddynrel        func(*LSym, *Reloc)
	Adddynsym        func(*Link, *LSym)
	Archinit         func()
	Archreloc        func(*Reloc, *LSym, *int64) int
	Archrelocvariant func(*Reloc, *LSym, int64) int64
	Asmb             func()
	Elfreloc1        func(*Reloc, int64) int
	Elfsetupplt      func()
	Gentext          func()
	Machoreloc1      func(*Reloc, int64) int
	Lput             func(uint32)
	Wput             func(uint16)
	Vput             func(uint64)
}

var Thearch Arch

var datap *LSym

var Debug [128]int

var literal string

var Lcsize int32

var rpath string

var Spsize int32

var symlist *LSym

var Symsize int32

// Terrible but standard terminology.
// A segment describes a block of file to load into memory.
// A section further describes the pieces of that block for
// use in debuggers and such.

const (
	MAXIO   = 8192
	MINFUNC = 16 // minimum size for a function
)

type Segment struct {
	Rwx     uint8  // permission as usual unix bits (5 = r-x etc)
	Vaddr   uint64 // virtual address
	Length  uint64 // length in memory
	Fileoff uint64 // file offset
	Filelen uint64 // length on disk
	Sect    *Section
}

type Section struct {
	Rwx     uint8
	Extnum  int16
	Align   int32
	Name    string
	Vaddr   uint64
	Length  uint64
	Next    *Section
	Seg     *Segment
	Elfsect interface{}
	Reloff  uint64
	Rellen  uint64
}

var Thestring string

var Thelinkarch *LinkArch

var outfile string

var ndynexp int

var dynexp []*LSym

var nldflag int

var ldflag []string

var havedynamic int

var Funcalign int

var iscgo bool

var elfglobalsymndx int

var flag_installsuffix string

var flag_race int

var Flag_shared int

var tracksym string

var interpreter string

var tmpdir string

var extld string

var extldflags string

var debug_s int // backup old value of debug['s']

var Ctxt *Link

var HEADR int32

var HEADTYPE int32

var INITRND int32

var INITTEXT int64

var INITDAT int64

var INITENTRY string /* entry point */

var nerrors int

var Linkmode int

var liveness int64

// for dynexport field of LSym
const (
	CgoExportDynamic = 1 << 0
	CgoExportStatic  = 1 << 1
)

var Segtext Segment

var Segrodata Segment

var Segdata Segment

var Segdwarf Segment

type Endian struct {
	e16 func([]byte) uint16
	e32 func([]byte) uint32
	e64 func([]byte) uint64
}

/* set by call to mywhatsys() */

/* whence for ldpkg */
const (
	FileObj = 0 + iota
	ArchiveObj
	Pkgdef
)

var headstring string

// buffered output

var Bso Biobuf

var coutbuf Biobuf

const (
	// Whether to assume that the external linker is "gold"
	// (http://sourceware.org/ml/binutils/2008-03/msg00162.html).
	AssumeGoldLinker = 0
)

const (
	symname = "__.GOSYMDEF"
	pkgname = "__.PKGDEF"
)

var cout *os.File

var version int

// Set if we see an object compiled by the host compiler that is not
// from a package that is known to support internal linking mode.
var externalobj = false

var goroot string

var goarch string

var goos string

var theline string

func Lflag(arg string) {
	Ctxt.Libdir = append(Ctxt.Libdir, arg)
}

/*
 * Unix doesn't like it when we write to a running (or, sometimes,
 * recently run) binary, so remove the output file before writing it.
 * On Windows 7, remove() can force a subsequent create() to fail.
 * S_ISREG() does not exist on Plan 9.
 */
func mayberemoveoutfile() {
	if fi, err := os.Lstat(outfile); err == nil && !fi.Mode().IsRegular() {
		return
	}
	os.Remove(outfile)
}

func libinit() {
	Funcalign = Thearch.Funcalign
	mywhatsys() // get goroot, goarch, goos

	// add goroot to the end of the libdir list.
	suffix := ""

	suffixsep := ""
	if flag_installsuffix != "" {
		suffixsep = "_"
		suffix = flag_installsuffix
	} else if flag_race != 0 {
		suffixsep = "_"
		suffix = "race"
	}

	Lflag(fmt.Sprintf("%s/pkg/%s_%s%s%s", goroot, goos, goarch, suffixsep, suffix))

	mayberemoveoutfile()
	f, err := os.OpenFile(outfile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Diag("cannot create %s: %v", outfile, err)
		Errorexit()
	}

	cout = f
	coutbuf = *Binitw(f)

	if INITENTRY == "" {
		if Flag_shared == 0 {
			INITENTRY = fmt.Sprintf("_rt0_%s_%s", goarch, goos)
		} else {
			INITENTRY = fmt.Sprintf("_rt0_%s_%s_lib", goarch, goos)
		}
	}

	Linklookup(Ctxt, INITENTRY, 0).Type = SXREF
}

func Errorexit() {
	if cout != nil {
		// For rmtemp run at atexit time on Windows.
		cout.Close()
	}

	if nerrors != 0 {
		if cout != nil {
			mayberemoveoutfile()
		}
		Exit(2)
	}

	Exit(0)
}

func loadinternal(name string) {
	var pname string

	found := 0
	for i := 0; i < len(Ctxt.Libdir); i++ {
		pname = fmt.Sprintf("%s/%s.a", Ctxt.Libdir[i], name)
		if Debug['v'] != 0 {
			fmt.Fprintf(&Bso, "searching for %s.a in %s\n", name, pname)
		}
		if obj.Access(pname, obj.AEXIST) >= 0 {
			addlibpath(Ctxt, "internal", "internal", pname, name)
			found = 1
			break
		}
	}

	if found == 0 {
		fmt.Fprintf(&Bso, "warning: unable to find %s.a\n", name)
	}
}

func loadlib() {
	if Flag_shared != 0 {
		s := Linklookup(Ctxt, "runtime.islibrary", 0)
		s.Dupok = 1
		Adduint8(Ctxt, s, 1)
	}

	loadinternal("runtime")
	if Thearch.Thechar == '5' {
		loadinternal("math")
	}
	if flag_race != 0 {
		loadinternal("runtime/race")
	}

	var i int
	for i = 0; i < len(Ctxt.Library); i++ {
		if Debug['v'] > 1 {
			fmt.Fprintf(&Bso, "%5.2f autolib: %s (from %s)\n", obj.Cputime(), Ctxt.Library[i].File, Ctxt.Library[i].Objref)
		}
		iscgo = iscgo || Ctxt.Library[i].Pkg == "runtime/cgo"
		objfile(Ctxt.Library[i].File, Ctxt.Library[i].Pkg)
	}

	if Linkmode == LinkAuto {
		if iscgo && externalobj {
			Linkmode = LinkExternal
		} else {
			Linkmode = LinkInternal
		}

		// Force external linking for android.
		if goos == "android" {
			Linkmode = LinkExternal
		}

		// cgo on Darwin must use external linking
		// we can always use external linking, but then there will be circular
		// dependency problems when compiling natively (external linking requires
		// runtime/cgo, runtime/cgo requires cmd/cgo, but cmd/cgo needs to be
		// compiled using external linking.)
		if Thearch.Thechar == '5' && HEADTYPE == Hdarwin && iscgo {
			Linkmode = LinkExternal
		}
	}

	if Linkmode == LinkExternal && !iscgo {
		// This indicates a user requested -linkmode=external.
		// The startup code uses an import of runtime/cgo to decide
		// whether to initialize the TLS.  So give it one.  This could
		// be handled differently but it's an unusual case.
		loadinternal("runtime/cgo")

		if i < len(Ctxt.Library) {
			objfile(Ctxt.Library[i].File, Ctxt.Library[i].Pkg)
		}

		// Pretend that we really imported the package.
		s := Linklookup(Ctxt, "go.importpath.runtime/cgo.", 0)

		s.Type = SDATA
		s.Dupok = 1
		s.Reachable = true

		// Provided by the code that imports the package.
		// Since we are simulating the import, we have to provide this string.
		cgostrsym := "go.string.\"runtime/cgo\""

		if Linkrlookup(Ctxt, cgostrsym, 0) == nil {
			s := Linklookup(Ctxt, cgostrsym, 0)
			s.Type = SRODATA
			s.Reachable = true
			addstrdata(cgostrsym, "runtime/cgo")
		}
	}

	if Linkmode == LinkInternal {
		// Drop all the cgo_import_static declarations.
		// Turns out we won't be needing them.
		for s := Ctxt.Allsym; s != nil; s = s.Allsym {
			if s.Type == SHOSTOBJ {
				// If a symbol was marked both
				// cgo_import_static and cgo_import_dynamic,
				// then we want to make it cgo_import_dynamic
				// now.
				if s.Extname != "" && s.Dynimplib != "" && s.Cgoexport == 0 {
					s.Type = SDYNIMPORT
				} else {
					s.Type = 0
				}
			}
		}
	}

	tlsg := Linklookup(Ctxt, "runtime.tlsg", 0)

	// For most ports, runtime.tlsg is a placeholder symbol for TLS
	// relocation. However, the Android and Darwin arm ports need it
	// to be a real variable.
	//
	// TODO(crawshaw): android should require leaving the tlsg->type
	// alone (as the runtime-provided SNOPTRBSS) just like darwin/arm.
	// But some other part of the linker is expecting STLSBSS.
	if goos != "darwin" || Thearch.Thechar != '5' {
		tlsg.Type = STLSBSS
	}
	tlsg.Size = int64(Thearch.Ptrsize)
	tlsg.Hide = 1
	tlsg.Reachable = true
	Ctxt.Tlsg = tlsg

	// Now that we know the link mode, trim the dynexp list.
	x := CgoExportDynamic

	if Linkmode == LinkExternal {
		x = CgoExportStatic
	}
	w := 0
	for i := 0; i < len(dynexp); i++ {
		if int(dynexp[i].Cgoexport)&x != 0 {
			dynexp[w] = dynexp[i]
			w++
		}
	}
	dynexp = dynexp[:w]

	// In internal link mode, read the host object files.
	if Linkmode == LinkInternal {
		hostobjs()
	} else {
		hostlinksetup()
	}

	// We've loaded all the code now.
	// If there are no dynamic libraries needed, gcc disables dynamic linking.
	// Because of this, glibc's dynamic ELF loader occasionally (like in version 2.13)
	// assumes that a dynamic binary always refers to at least one dynamic library.
	// Rather than be a source of test cases for glibc, disable dynamic linking
	// the same way that gcc would.
	//
	// Exception: on OS X, programs such as Shark only work with dynamic
	// binaries, so leave it enabled on OS X (Mach-O) binaries.
	// Also leave it enabled on Solaris which doesn't support
	// statically linked binaries.
	if Flag_shared == 0 && havedynamic == 0 && HEADTYPE != Hdarwin && HEADTYPE != Hsolaris {
		Debug['d'] = 1
	}

	importcycles()
}

/*
 * look for the next file in an archive.
 * adapted from libmach.
 */
func nextar(bp *Biobuf, off int64, a *ArHdr) int64 {
	if off&1 != 0 {
		off++
	}
	Bseek(bp, off, 0)
	buf := make([]byte, SAR_HDR)
	if n := Bread(bp, buf); n < len(buf) {
		if n >= 0 {
			return 0
		}
		return -1
	}

	a.name = artrim(buf[0:16])
	a.date = artrim(buf[16:28])
	a.uid = artrim(buf[28:34])
	a.gid = artrim(buf[34:40])
	a.mode = artrim(buf[40:48])
	a.size = artrim(buf[48:58])
	a.fmag = artrim(buf[58:60])

	arsize := atolwhex(a.size)
	if arsize&1 != 0 {
		arsize++
	}
	return int64(arsize) + SAR_HDR
}

func objfile(file string, pkg string) {
	pkg = pathtoprefix(pkg)

	if Debug['v'] > 1 {
		fmt.Fprintf(&Bso, "%5.2f ldobj: %s (%s)\n", obj.Cputime(), file, pkg)
	}
	Bflush(&Bso)
	var err error
	var f *Biobuf
	f, err = Bopenr(file)
	if err != nil {
		Diag("cannot open file %s: %v", file, err)
		Errorexit()
	}

	magbuf := make([]byte, len(ARMAG))
	if Bread(f, magbuf) != len(magbuf) || !strings.HasPrefix(string(magbuf), ARMAG) {
		/* load it as a regular file */
		l := Bseek(f, 0, 2)

		Bseek(f, 0, 0)
		ldobj(f, pkg, l, file, file, FileObj)
		Bterm(f)

		return
	}

	/* skip over optional __.GOSYMDEF and process __.PKGDEF */
	off := Boffset(f)

	var arhdr ArHdr
	l := nextar(f, off, &arhdr)
	var pname string
	if l <= 0 {
		Diag("%s: short read on archive file symbol header", file)
		goto out
	}

	if strings.HasPrefix(arhdr.name, symname) {
		off += l
		l = nextar(f, off, &arhdr)
		if l <= 0 {
			Diag("%s: short read on archive file symbol header", file)
			goto out
		}
	}

	if !strings.HasPrefix(arhdr.name, pkgname) {
		Diag("%s: cannot find package header", file)
		goto out
	}

	off += l

	if Debug['u'] != 0 {
		ldpkg(f, pkg, atolwhex(arhdr.size), file, Pkgdef)
	}

	/*
	 * load all the object files from the archive now.
	 * this gives us sequential file access and keeps us
	 * from needing to come back later to pick up more
	 * objects.  it breaks the usual C archive model, but
	 * this is Go, not C.  the common case in Go is that
	 * we need to load all the objects, and then we throw away
	 * the individual symbols that are unused.
	 *
	 * loading every object will also make it possible to
	 * load foreign objects not referenced by __.GOSYMDEF.
	 */
	for {
		l = nextar(f, off, &arhdr)
		if l == 0 {
			break
		}
		if l < 0 {
			Diag("%s: malformed archive", file)
			Errorexit()
			goto out
		}

		off += l

		pname = fmt.Sprintf("%s(%s)", file, arhdr.name)
		l = atolwhex(arhdr.size)
		ldobj(f, pkg, l, pname, file, ArchiveObj)
	}

out:
	Bterm(f)
}

type Hostobj struct {
	ld     func(*Biobuf, string, int64, string)
	pkg    string
	pn     string
	file   string
	off    int64
	length int64
}

var hostobj []Hostobj

var nhostobj int

var mhostobj int

// These packages can use internal linking mode.
// Others trigger external mode.
var internalpkg = []string{
	"crypto/x509",
	"net",
	"os/user",
	"runtime/cgo",
	"runtime/race",
}

func ldhostobj(ld func(*Biobuf, string, int64, string), f *Biobuf, pkg string, length int64, pn string, file string) {
	isinternal := false
	for i := 0; i < len(internalpkg); i++ {
		if pkg == internalpkg[i] {
			isinternal = true
			break
		}
	}

	// DragonFly declares errno with __thread, which results in a symbol
	// type of R_386_TLS_GD or R_X86_64_TLSGD. The Go linker does not
	// currently know how to handle TLS relocations, hence we have to
	// force external linking for any libraries that link in code that
	// uses errno. This can be removed if the Go linker ever supports
	// these relocation types.
	if HEADTYPE == Hdragonfly {
		if pkg == "net" || pkg == "os/user" {
			isinternal = false
		}
	}

	if !isinternal {
		externalobj = true
	}

	hostobj = append(hostobj, Hostobj{})
	h := &hostobj[len(hostobj)-1]
	h.ld = ld
	h.pkg = pkg
	h.pn = pn
	h.file = file
	h.off = Boffset(f)
	h.length = length
}

func hostobjs() {
	var f *Biobuf
	var h *Hostobj

	for i := 0; i < len(hostobj); i++ {
		h = &hostobj[i]
		var err error
		f, err = Bopenr(h.file)
		if f == nil {
			Ctxt.Cursym = nil
			Diag("cannot reopen %s: %v", h.pn, err)
			Errorexit()
		}

		Bseek(f, h.off, 0)
		h.ld(f, h.pkg, h.length, h.pn)
		Bterm(f)
	}
}

// provided by lib9

func rmtemp() {
	os.RemoveAll(tmpdir)
}

func hostlinksetup() {
	if Linkmode != LinkExternal {
		return
	}

	// create temporary directory and arrange cleanup
	if tmpdir == "" {
		dir, err := ioutil.TempDir("", "go-link-")
		if err != nil {
			log.Fatal(err)
		}
		tmpdir = dir
		AtExit(rmtemp)
	}

	// change our output to temporary object file
	cout.Close()

	p := fmt.Sprintf("%s/go.o", tmpdir)
	var err error
	cout, err = os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0775)
	if err != nil {
		Diag("cannot create %s: %v", p, err)
		Errorexit()
	}

	coutbuf = *Binitw(cout)
}

var hostlink_buf = make([]byte, 64*1024)

func hostlink() {
	if Linkmode != LinkExternal || nerrors > 0 {
		return
	}

	if extld == "" {
		extld = "gcc"
	}
	var argv []string
	argv = append(argv, extld)
	switch Thearch.Thechar {
	case '8':
		argv = append(argv, "-m32")

	case '6',
		'9':
		argv = append(argv, "-m64")

	case '5':
		argv = append(argv, "-marm")
	}

	if Debug['s'] == 0 && debug_s == 0 {
		argv = append(argv, "-gdwarf-2")
	} else {
		argv = append(argv, "-s")
	}

	if HEADTYPE == Hdarwin {
		argv = append(argv, "-Wl,-no_pie,-pagezero_size,4000000")
	}
	if HEADTYPE == Hopenbsd {
		argv = append(argv, "-Wl,-nopie")
	}

	if Iself && AssumeGoldLinker != 0 /*TypeKind(100016)*/ {
		argv = append(argv, "-Wl,--rosegment")
	}

	if Flag_shared != 0 {
		argv = append(argv, "-Wl,-Bsymbolic")
		argv = append(argv, "-shared")
	}

	argv = append(argv, "-o")
	argv = append(argv, outfile)

	if rpath != "" {
		argv = append(argv, fmt.Sprintf("-Wl,-rpath,%s", rpath))
	}

	// Force global symbols to be exported for dlopen, etc.
	if Iself {
		argv = append(argv, "-rdynamic")
	}

	if strings.Contains(argv[0], "clang") {
		argv = append(argv, "-Qunused-arguments")
	}

	// already wrote main object file
	// copy host objects to temporary directory
	var f *Biobuf
	var h *Hostobj
	var length int
	var n int
	var p string
	for i := 0; i < len(hostobj); i++ {
		h = &hostobj[i]
		var err error
		f, err = Bopenr(h.file)
		if f == nil {
			Ctxt.Cursym = nil
			Diag("cannot reopen %s: %v", h.pn, err)
			Errorexit()
		}

		Bseek(f, h.off, 0)
		p = fmt.Sprintf("%s/%06d.o", tmpdir, i)
		argv = append(argv, p)
		w, err := os.Create(p)
		if err != nil {
			Ctxt.Cursym = nil
			Diag("cannot create %s: %v", p, err)
			Errorexit()
		}

		length = int(h.length)
		for length > 0 {
			n = Bread(f, hostlink_buf)
			if n <= 0 {
				break
			}
			if n > length {
				n = length
			}
			if _, err = w.Write(hostlink_buf[:n]); err != nil {
				log.Fatal(err)
			}
			length -= n
		}

		if err := w.Close(); err != nil {
			Ctxt.Cursym = nil
			Diag("cannot write %s: %v", p, err)
			Errorexit()
		}

		Bterm(f)
	}

	argv = append(argv, fmt.Sprintf("%s/go.o", tmpdir))
	var i int
	for i = 0; i < len(ldflag); i++ {
		argv = append(argv, ldflag[i])
	}

	for _, p = range strings.Fields(extldflags) {
		argv = append(argv, p)

		// clang, unlike GCC, passes -rdynamic to the linker
		// even when linking with -static, causing a linker
		// error when using GNU ld.  So take out -rdynamic if
		// we added it.  We do it in this order, rather than
		// only adding -rdynamic later, so that -extldflags
		// can override -rdynamic without using -static.
		if Iself && p == "-static" {
			for i = range argv {
				if argv[i] == "-rdynamic" {
					argv[i] = "-static"
				}
			}
		}
	}

	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "host link:")
		for i = range argv {
			fmt.Fprintf(&Bso, " %v", plan9quote(argv[i]))
		}
		fmt.Fprintf(&Bso, "\n")
		Bflush(&Bso)
	}

	if out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput(); err != nil {
		Ctxt.Cursym = nil
		Diag("%s: running %s failed: %v\n%s", os.Args[0], argv[0], err, out)
		Errorexit()
	}
}

func ldobj(f *Biobuf, pkg string, length int64, pn string, file string, whence int) {
	eof := Boffset(f) + length

	pn = pn

	start := Boffset(f)
	c1 := Bgetc(f)
	c2 := Bgetc(f)
	c3 := Bgetc(f)
	c4 := Bgetc(f)
	Bseek(f, start, 0)

	magic := uint32(c1)<<24 | uint32(c2)<<16 | uint32(c3)<<8 | uint32(c4)
	if magic == 0x7f454c46 { // \x7F E L F
		ldhostobj(ldelf, f, pkg, length, pn, file)
		return
	}

	if magic&^1 == 0xfeedface || magic&^0x01000000 == 0xcefaedfe {
		ldhostobj(ldmacho, f, pkg, length, pn, file)
		return
	}

	if c1 == 0x4c && c2 == 0x01 || c1 == 0x64 && c2 == 0x86 {
		ldhostobj(ldpe, f, pkg, length, pn, file)
		return
	}

	/* check the header */
	line := Brdline(f, '\n')

	var import0 int64
	var import1 int64
	var t string
	if line == "" {
		if Blinelen(f) > 0 {
			Diag("%s: not an object file", pn)
			return
		}

		goto eof
	}

	if !strings.HasPrefix(line, "go object ") {
		if strings.HasSuffix(pn, ".go") {
			fmt.Printf("%cl: input %s is not .%c file (use %cg to compile .go files)\n", Thearch.Thechar, pn, Thearch.Thechar, Thearch.Thechar)
			Errorexit()
		}

		if line == Thestring {
			// old header format: just $GOOS
			Diag("%s: stale object file", pn)

			return
		}

		Diag("%s: not an object file", pn)

		return
	}

	// First, check that the basic goos, goarch, and version match.
	t = fmt.Sprintf("%s %s %s ", goos, obj.Getgoarch(), obj.Getgoversion())

	line = strings.TrimRight(line, "\n")
	if !strings.HasPrefix(line[10:]+" ", t) && Debug['f'] == 0 {
		Diag("%s: object is [%s] expected [%s]", pn, line[10:], t)
		return
	}

	// Second, check that longer lines match each other exactly,
	// so that the Go compiler and write additional information
	// that must be the same from run to run.
	if len(line) >= len(t)+10 {
		if theline == "" {
			theline = line[10:]
		} else if theline != line[10:] {
			Diag("%s: object is [%s] expected [%s]", pn, line[10:], theline)
			return
		}
	}

	/* skip over exports and other info -- ends with \n!\n */
	import0 = Boffset(f)

	c1 = '\n' // the last line ended in \n
	c2 = Bgetc(f)
	c3 = Bgetc(f)
	for c1 != '\n' || c2 != '!' || c3 != '\n' {
		c1 = c2
		c2 = c3
		c3 = Bgetc(f)
		if c3 == Beof {
			goto eof
		}
	}

	import1 = Boffset(f)

	Bseek(f, import0, 0)
	ldpkg(f, pkg, import1-import0-2, pn, whence) // -2 for !\n
	Bseek(f, import1, 0)

	ldobjfile(Ctxt, f, pkg, eof-Boffset(f), pn)

	return

eof:
	Diag("truncated object file: %s", pn)
}

func zerosig(sp string) {
	s := Linklookup(Ctxt, sp, 0)
	s.Sig = 0
}

func mywhatsys() {
	goroot = obj.Getgoroot()
	goos = obj.Getgoos()
	goarch = obj.Getgoarch()

	if !strings.HasPrefix(goarch, Thestring) {
		log.Fatalf("cannot use %cc with GOARCH=%s", Thearch.Thechar, goarch)
	}
}

func pathchar() int {
	return '/'
}

var hunk []byte

var nhunk uint32

const (
	NHUNK = 10 << 20
)

// Copied from ../gc/subr.c:/^pathtoprefix; must stay in sync.
/*
 * Convert raw string to the prefix that will be used in the symbol table.
 * Invalid bytes turn into %xx.	 Right now the only bytes that need
 * escaping are %, ., and ", but we escape all control characters too.
 *
 * If you edit this, edit ../gc/subr.c:/^pathtoprefix too.
 * If you edit this, edit ../../debug/goobj/read.go:/importPathToPrefix too.
 */
func pathtoprefix(s string) string {
	slash := strings.LastIndex(s, "/")
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c <= ' ' || i >= slash && c == '.' || c == '%' || c == '"' || c >= 0x7F {
			var buf bytes.Buffer
			for i := 0; i < len(s); i++ {
				c := s[i]
				if c <= ' ' || i >= slash && c == '.' || c == '%' || c == '"' || c >= 0x7F {
					fmt.Fprintf(&buf, "%%%02x", c)
					continue
				}
				buf.WriteByte(c)
			}
			return buf.String()
		}
	}
	return s
}

func iconv(p string) string {
	if p == "" {
		var fp string
		fp += "<nil>"
		return fp
	}

	p = pathtoprefix(p)
	var fp string
	fp += p
	return fp
}

func addsection(seg *Segment, name string, rwx int) *Section {
	var l **Section

	for l = &seg.Sect; *l != nil; l = &(*l).Next {
	}
	sect := new(Section)
	sect.Rwx = uint8(rwx)
	sect.Name = name
	sect.Seg = seg
	sect.Align = int32(Thearch.Ptrsize) // everything is at least pointer-aligned
	*l = sect
	return sect
}

func Le16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8
}

func Le32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func Le64(b []byte) uint64 {
	return uint64(Le32(b)) | uint64(Le32(b[4:]))<<32
}

func Be16(b []byte) uint16 {
	return uint16(b[0])<<8 | uint16(b[1])
}

func Be32(b []byte) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

func Be64(b []byte) uint64 {
	return uint64(Be32(b))<<32 | uint64(Be32(b[4:]))
}

var be = Endian{Be16, Be32, Be64}

var le = Endian{Le16, Le32, Le64}

type Chain struct {
	sym   *LSym
	up    *Chain
	limit int // limit on entry to sym
}

var morestack *LSym

var newstack *LSym

// TODO: Record enough information in new object files to
// allow stack checks here.

func haslinkregister() bool {
	return Thearch.Thechar == '5' || Thearch.Thechar == '9'
}

func callsize() int {
	if haslinkregister() {
		return 0
	}
	return Thearch.Regsize
}

func dostkcheck() {
	var ch Chain

	morestack = Linklookup(Ctxt, "runtime.morestack", 0)
	newstack = Linklookup(Ctxt, "runtime.newstack", 0)

	// Every splitting function ensures that there are at least StackLimit
	// bytes available below SP when the splitting prologue finishes.
	// If the splitting function calls F, then F begins execution with
	// at least StackLimit - callsize() bytes available.
	// Check that every function behaves correctly with this amount
	// of stack, following direct calls in order to piece together chains
	// of non-splitting functions.
	ch.up = nil

	ch.limit = obj.StackLimit - callsize()

	// Check every function, but do the nosplit functions in a first pass,
	// to make the printed failure chains as short as possible.
	for s := Ctxt.Textp; s != nil; s = s.Next {
		// runtime.racesymbolizethunk is called from gcc-compiled C
		// code running on the operating system thread stack.
		// It uses more than the usual amount of stack but that's okay.
		if s.Name == "runtime.racesymbolizethunk" {
			continue
		}

		if s.Nosplit != 0 {
			Ctxt.Cursym = s
			ch.sym = s
			stkcheck(&ch, 0)
		}
	}

	for s := Ctxt.Textp; s != nil; s = s.Next {
		if s.Nosplit == 0 {
			Ctxt.Cursym = s
			ch.sym = s
			stkcheck(&ch, 0)
		}
	}
}

func stkcheck(up *Chain, depth int) int {
	limit := up.limit
	s := up.sym

	// Don't duplicate work: only need to consider each
	// function at top of safe zone once.
	if limit == obj.StackLimit-callsize() {
		if s.Stkcheck != 0 {
			return 0
		}
		s.Stkcheck = 1
	}

	if depth > 100 {
		Diag("nosplit stack check too deep")
		stkbroke(up, 0)
		return -1
	}

	if s.External != 0 || s.Pcln == nil {
		// external function.
		// should never be called directly.
		// only diagnose the direct caller.
		if depth == 1 && s.Type != SXREF {
			Diag("call to external function %s", s.Name)
		}
		return -1
	}

	if limit < 0 {
		stkbroke(up, limit)
		return -1
	}

	// morestack looks like it calls functions,
	// but it switches the stack pointer first.
	if s == morestack {
		return 0
	}

	var ch Chain
	ch.up = up

	// Walk through sp adjustments in function, consuming relocs.
	ri := 0

	endr := len(s.R)
	var ch1 Chain
	var pcsp Pciter
	var r *Reloc
	for pciterinit(Ctxt, &pcsp, &s.Pcln.Pcsp); pcsp.done == 0; pciternext(&pcsp) {
		// pcsp.value is in effect for [pcsp.pc, pcsp.nextpc).

		// Check stack size in effect for this span.
		if int32(limit)-pcsp.value < 0 {
			stkbroke(up, int(int32(limit)-pcsp.value))
			return -1
		}

		// Process calls in this span.
		for ; ri < endr && uint32(s.R[ri].Off) < pcsp.nextpc; ri++ {
			r = &s.R[ri]
			switch r.Type {
			// Direct call.
			case R_CALL,
				R_CALLARM,
				R_CALLPOWER:
				ch.limit = int(int32(limit) - pcsp.value - int32(callsize()))

				ch.sym = r.Sym
				if stkcheck(&ch, depth+1) < 0 {
					return -1
				}

				// If this is a call to morestack, we've just raised our limit back
				// to StackLimit beyond the frame size.
				if strings.HasPrefix(r.Sym.Name, "runtime.morestack") {
					limit = int(obj.StackLimit + s.Locals)
					if haslinkregister() {
						limit += Thearch.Regsize
					}
				}

				// Indirect call.  Assume it is a call to a splitting function,
			// so we have to make sure it can call morestack.
			// Arrange the data structures to report both calls, so that
			// if there is an error, stkprint shows all the steps involved.
			case R_CALLIND:
				ch.limit = int(int32(limit) - pcsp.value - int32(callsize()))

				ch.sym = nil
				ch1.limit = ch.limit - callsize() // for morestack in called prologue
				ch1.up = &ch
				ch1.sym = morestack
				if stkcheck(&ch1, depth+2) < 0 {
					return -1
				}
			}
		}
	}

	return 0
}

func stkbroke(ch *Chain, limit int) {
	Diag("nosplit stack overflow")
	stkprint(ch, limit)
}

func stkprint(ch *Chain, limit int) {
	var name string

	if ch.sym != nil {
		name = ch.sym.Name
	} else {
		name = "function pointer"
	}

	if ch.up == nil {
		// top of chain.  ch->sym != nil.
		if ch.sym.Nosplit != 0 {
			fmt.Printf("\t%d\tassumed on entry to %s\n", ch.limit, name)
		} else {
			fmt.Printf("\t%d\tguaranteed after split check in %s\n", ch.limit, name)
		}
	} else {
		stkprint(ch.up, ch.limit+callsize())
		if !haslinkregister() {
			fmt.Printf("\t%d\ton entry to %s\n", ch.limit, name)
		}
	}

	if ch.limit != limit {
		fmt.Printf("\t%d\tafter %s uses %d\n", limit, name, ch.limit-limit)
	}
}

func Yconv(s *LSym) string {
	var fp string

	if s == nil {
		fp += fmt.Sprintf("<nil>")
	} else {
		fmt_ := ""
		fmt_ += fmt.Sprintf("%s @0x%08x [%d]", s.Name, int64(s.Value), int64(s.Size))
		for i := 0; int64(i) < s.Size; i++ {
			if i%8 == 0 {
				fmt_ += fmt.Sprintf("\n\t0x%04x ", i)
			}
			fmt_ += fmt.Sprintf("%02x ", s.P[i])
		}

		fmt_ += fmt.Sprintf("\n")
		for i := 0; i < len(s.R); i++ {
			fmt_ += fmt.Sprintf("\t0x%04x[%x] %d %s[%x]\n", s.R[i].Off, s.R[i].Siz, s.R[i].Type, s.R[i].Sym.Name, int64(s.R[i].Add))
		}

		str := fmt_
		fp += str
	}

	return fp
}

func Cflush() {
	Bflush(&coutbuf)
}

func Cpos() int64 {
	return Boffset(&coutbuf)
}

func Cseek(p int64) {
	Bseek(&coutbuf, p, 0)
}

func Cwrite(p []byte) {
	Bwrite(&coutbuf, p)
}

func Cput(c uint8) {
	Bputc(&coutbuf, c)
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: %cl [options] obj.%c\n", Thearch.Thechar, Thearch.Thechar)
	obj.Flagprint(2)
	Exit(2)
}

func setheadtype(s string) {
	h := headtype(s)
	if h < 0 {
		fmt.Fprintf(os.Stderr, "unknown header type -H %s\n", s)
		Errorexit()
	}

	headstring = s
	HEADTYPE = int32(headtype(s))
}

func setinterp(s string) {
	Debug['I'] = 1 // denote cmdline interpreter override
	interpreter = s
}

func doversion() {
	fmt.Printf("%cl version %s\n", Thearch.Thechar, obj.Getgoversion())
	Errorexit()
}

func genasmsym(put func(*LSym, string, int, int64, int64, int, *LSym)) {
	// These symbols won't show up in the first loop below because we
	// skip STEXT symbols. Normal STEXT symbols are emitted by walking textp.
	s := Linklookup(Ctxt, "runtime.text", 0)

	if s.Type == STEXT {
		put(s, s.Name, 'T', s.Value, s.Size, int(s.Version), nil)
	}
	s = Linklookup(Ctxt, "runtime.etext", 0)
	if s.Type == STEXT {
		put(s, s.Name, 'T', s.Value, s.Size, int(s.Version), nil)
	}

	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if s.Hide != 0 || (s.Name[0] == '.' && s.Version == 0 && s.Name != ".rathole") {
			continue
		}
		switch s.Type & SMASK {
		case SCONST,
			SRODATA,
			SSYMTAB,
			SPCLNTAB,
			SDATA,
			SNOPTRDATA,
			SELFROSECT,
			SMACHOGOT,
			STYPE,
			SSTRING,
			SGOSTRING,
			SWINDOWS:
			if !s.Reachable {
				continue
			}
			put(s, s.Name, 'D', Symaddr(s), s.Size, int(s.Version), s.Gotype)
			continue

		case SBSS,
			SNOPTRBSS:
			if !s.Reachable {
				continue
			}
			if len(s.P) > 0 {
				Diag("%s should not be bss (size=%d type=%d special=%d)", s.Name, int(len(s.P)), s.Type, s.Special)
			}
			put(s, s.Name, 'B', Symaddr(s), s.Size, int(s.Version), s.Gotype)
			continue

		case SFILE:
			put(nil, s.Name, 'f', s.Value, 0, int(s.Version), nil)
			continue
		}
	}

	var a *Auto
	var off int32
	for s := Ctxt.Textp; s != nil; s = s.Next {
		put(s, s.Name, 'T', s.Value, s.Size, int(s.Version), s.Gotype)

		// NOTE(ality): acid can't produce a stack trace without .frame symbols
		put(nil, ".frame", 'm', int64(s.Locals)+int64(Thearch.Ptrsize), 0, 0, nil)

		for a = s.Autom; a != nil; a = a.Link {
			// Emit a or p according to actual offset, even if label is wrong.
			// This avoids negative offsets, which cannot be encoded.
			if a.Name != A_AUTO && a.Name != A_PARAM {
				continue
			}

			// compute offset relative to FP
			if a.Name == A_PARAM {
				off = a.Aoffset
			} else {
				off = a.Aoffset - int32(Thearch.Ptrsize)
			}

			// FP
			if off >= 0 {
				put(nil, a.Asym.Name, 'p', int64(off), 0, 0, a.Gotype)
				continue
			}

			// SP
			if off <= int32(-Thearch.Ptrsize) {
				put(nil, a.Asym.Name, 'a', -(int64(off) + int64(Thearch.Ptrsize)), 0, 0, a.Gotype)
				continue
			}
		}
	}

	// Otherwise, off is addressing the saved program counter.
	// Something underhanded is going on. Say nothing.
	if Debug['v'] != 0 || Debug['n'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f symsize = %d\n", obj.Cputime(), uint32(Symsize))
	}
	Bflush(&Bso)
}

func Symaddr(s *LSym) int64 {
	if !s.Reachable {
		Diag("unreachable symbol in symaddr - %s", s.Name)
	}
	return s.Value
}

func xdefine(p string, t int, v int64) {
	s := Linklookup(Ctxt, p, 0)
	s.Type = int16(t)
	s.Value = v
	s.Reachable = true
	s.Special = 1
}

func datoff(addr int64) int64 {
	if uint64(addr) >= Segdata.Vaddr {
		return int64(uint64(addr) - Segdata.Vaddr + Segdata.Fileoff)
	}
	if uint64(addr) >= Segtext.Vaddr {
		return int64(uint64(addr) - Segtext.Vaddr + Segtext.Fileoff)
	}
	Diag("datoff %#x", addr)
	return 0
}

func Entryvalue() int64 {
	a := INITENTRY
	if a[0] >= '0' && a[0] <= '9' {
		return atolwhex(a)
	}
	s := Linklookup(Ctxt, a, 0)
	if s.Type == 0 {
		return INITTEXT
	}
	if s.Type != STEXT {
		Diag("entry not text: %s", s.Name)
	}
	return s.Value
}

func undefsym(s *LSym) {
	var r *Reloc

	Ctxt.Cursym = s
	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		if r.Sym == nil { // happens for some external ARM relocs
			continue
		}
		if r.Sym.Type == Sxxx || r.Sym.Type == SXREF {
			Diag("undefined: %s", r.Sym.Name)
		}
		if !r.Sym.Reachable {
			Diag("use of unreachable symbol: %s", r.Sym.Name)
		}
	}
}

func undef() {
	for s := Ctxt.Textp; s != nil; s = s.Next {
		undefsym(s)
	}
	for s := datap; s != nil; s = s.Next {
		undefsym(s)
	}
	if nerrors > 0 {
		Errorexit()
	}
}

func callgraph() {
	if Debug['c'] == 0 {
		return
	}

	var i int
	var r *Reloc
	for s := Ctxt.Textp; s != nil; s = s.Next {
		for i = 0; i < len(s.R); i++ {
			r = &s.R[i]
			if r.Sym == nil {
				continue
			}
			if (r.Type == R_CALL || r.Type == R_CALLARM || r.Type == R_CALLPOWER) && r.Sym.Type == STEXT {
				fmt.Fprintf(&Bso, "%s calls %s\n", s.Name, r.Sym.Name)
			}
		}
	}
}

func Diag(format string, args ...interface{}) {
	tn := ""
	sep := ""
	if Ctxt.Cursym != nil {
		tn = Ctxt.Cursym.Name
		sep = ": "
	}
	fmt.Printf("%s%s%s\n", tn, sep, fmt.Sprintf(format, args...))

	nerrors++
	if nerrors > 20 {
		fmt.Printf("too many errors\n")
		Errorexit()
	}
}

func checkgo() {
	if Debug['C'] == 0 {
		return
	}

	// TODO(rsc,khr): Eventually we want to get to no Go-called C functions at all,
	// which would simplify this logic quite a bit.

	// Mark every Go-called C function with cfunc=2, recursively.
	var changed int
	var i int
	var r *Reloc
	var s *LSym
	for {
		changed = 0
		for s = Ctxt.Textp; s != nil; s = s.Next {
			if s.Cfunc == 0 || (s.Cfunc == 2 && s.Nosplit != 0) {
				for i = 0; i < len(s.R); i++ {
					r = &s.R[i]
					if r.Sym == nil {
						continue
					}
					if (r.Type == R_CALL || r.Type == R_CALLARM) && r.Sym.Type == STEXT {
						if r.Sym.Cfunc == 1 {
							changed = 1
							r.Sym.Cfunc = 2
						}
					}
				}
			}
		}
		if changed == 0 {
			break
		}
	}

	// Complain about Go-called C functions that can split the stack
	// (that can be preempted for garbage collection or trigger a stack copy).
	for s := Ctxt.Textp; s != nil; s = s.Next {
		if s.Cfunc == 0 || (s.Cfunc == 2 && s.Nosplit != 0) {
			for i = 0; i < len(s.R); i++ {
				r = &s.R[i]
				if r.Sym == nil {
					continue
				}
				if (r.Type == R_CALL || r.Type == R_CALLARM) && r.Sym.Type == STEXT {
					if s.Cfunc == 0 && r.Sym.Cfunc == 2 && r.Sym.Nosplit == 0 {
						fmt.Printf("Go %s calls C %s\n", s.Name, r.Sym.Name)
					} else if s.Cfunc == 2 && s.Nosplit != 0 && r.Sym.Nosplit == 0 {
						fmt.Printf("Go calls C %s calls %s\n", s.Name, r.Sym.Name)
					}
				}
			}
		}
	}
}

func Rnd(v int64, r int64) int64 {
	if r <= 0 {
		return v
	}
	v += r - 1
	c := v % r
	if c < 0 {
		c += r
	}
	v -= c
	return v
}
