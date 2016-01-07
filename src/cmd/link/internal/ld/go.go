// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

package ld

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"os"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

// replace all "". with pkg.
func expandpkg(t0 string, pkg string) string {
	return strings.Replace(t0, `"".`, pkg+".", -1)
}

// TODO:
//	generate debugging section in binary.
//	once the dust settles, try to move some code to
//		libmach, so that other linkers and ar can share.

func ldpkg(f *obj.Biobuf, pkg string, length int64, filename string, whence int) {
	var p0, p1 int

	if Debug['g'] != 0 {
		return
	}

	if int64(int(length)) != length {
		fmt.Fprintf(os.Stderr, "%s: too much pkg data in %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			errorexit()
		}
		return
	}

	// In a __.PKGDEF, we only care about the package name.
	// Don't read all the export data.
	if length > 1000 && whence == Pkgdef {
		length = 1000
	}

	bdata := make([]byte, length)
	if int64(obj.Bread(f, bdata)) != length {
		fmt.Fprintf(os.Stderr, "%s: short pkg read %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			errorexit()
		}
		return
	}
	data := string(bdata)

	// first \n$$ marks beginning of exports - skip rest of line
	p0 = strings.Index(data, "\n$$")
	if p0 < 0 {
		if Debug['u'] != 0 && whence != ArchiveObj {
			Exitf("cannot find export data in %s", filename)
		}
		return
	}

	// \n$$B marks the beginning of binary export data - don't skip over the B
	p0 += 3
	for p0 < len(data) && data[p0] != '\n' && data[p0] != 'B' {
		p0++
	}

	// second marks end of exports / beginning of local data
	p1 = strings.Index(data[p0:], "\n$$\n")
	if p1 < 0 && whence == Pkgdef {
		p1 = len(data) - p0
	}
	if p1 < 0 {
		fmt.Fprintf(os.Stderr, "%s: cannot find end of exports in %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			errorexit()
		}
		return
	}
	p1 += p0

	for p0 < p1 && data[p0] != 'B' && (data[p0] == ' ' || data[p0] == '\t' || data[p0] == '\n') {
		p0++
	}
	// don't check this section if we have binary (B) export data
	// TODO fix this eventually
	if p0 < p1 && data[p0] != 'B' {
		if !strings.HasPrefix(data[p0:], "package ") {
			fmt.Fprintf(os.Stderr, "%s: bad package section in %s - %.20s\n", os.Args[0], filename, data[p0:])
			if Debug['u'] != 0 {
				errorexit()
			}
			return
		}

		p0 += 8
		for p0 < p1 && (data[p0] == ' ' || data[p0] == '\t' || data[p0] == '\n') {
			p0++
		}
		pname := p0
		for p0 < p1 && data[p0] != ' ' && data[p0] != '\t' && data[p0] != '\n' {
			p0++
		}
		if Debug['u'] != 0 && whence != ArchiveObj && (p0+6 > p1 || !strings.HasPrefix(data[p0:], " safe\n")) {
			Exitf("load of unsafe package %s", filename)
		}

		name := data[pname:p0]
		for p0 < p1 && data[p0] != '\n' {
			p0++
		}
		if p0 < p1 {
			p0++
		}

		if pkg == "main" && name != "main" {
			Exitf("%s: not package main (package %s)", filename, name)
		}
	}

	// __.PKGDEF has no cgo section - those are in the C compiler-generated object files.
	if whence == Pkgdef {
		return
	}

	// look for cgo section
	p0 = strings.Index(data[p1:], "\n$$  // cgo")
	if p0 >= 0 {
		p0 += p1
		i := strings.IndexByte(data[p0+1:], '\n')
		if i < 0 {
			fmt.Fprintf(os.Stderr, "%s: found $$ // cgo but no newline in %s\n", os.Args[0], filename)
			if Debug['u'] != 0 {
				errorexit()
			}
			return
		}
		p0 += 1 + i

		p1 = strings.Index(data[p0:], "\n$$")
		if p1 < 0 {
			p1 = strings.Index(data[p0:], "\n!\n")
		}
		if p1 < 0 {
			fmt.Fprintf(os.Stderr, "%s: cannot find end of // cgo section in %s\n", os.Args[0], filename)
			if Debug['u'] != 0 {
				errorexit()
			}
			return
		}
		p1 += p0

		loadcgo(filename, pkg, data[p0:p1])
	}
}

func loadcgo(file string, pkg string, p string) {
	var next string
	var q string
	var f []string
	var local string
	var remote string
	var lib string
	var s *LSym

	p0 := ""
	for ; p != ""; p = next {
		if i := strings.Index(p, "\n"); i >= 0 {
			p, next = p[:i], p[i+1:]
		} else {
			next = ""
		}

		p0 = p // save for error message
		f = tokenize(p)
		if len(f) == 0 {
			continue
		}

		if f[0] == "cgo_import_dynamic" {
			if len(f) < 2 || len(f) > 4 {
				goto err
			}

			local = f[1]
			remote = local
			if len(f) > 2 {
				remote = f[2]
			}
			lib = ""
			if len(f) > 3 {
				lib = f[3]
			}

			if Debug['d'] != 0 {
				fmt.Fprintf(os.Stderr, "%s: %s: cannot use dynamic imports with -d flag\n", os.Args[0], file)
				nerrors++
				return
			}

			if local == "_" && remote == "_" {
				// allow #pragma dynimport _ _ "foo.so"
				// to force a link of foo.so.
				havedynamic = 1

				if HEADTYPE == obj.Hdarwin {
					Machoadddynlib(lib)
				} else {
					dynlib = append(dynlib, lib)
				}
				continue
			}

			local = expandpkg(local, pkg)
			q = ""
			if i := strings.Index(remote, "#"); i >= 0 {
				remote, q = remote[:i], remote[i+1:]
			}
			s = Linklookup(Ctxt, local, 0)
			if local != f[1] {
			}
			if s.Type == 0 || s.Type == obj.SXREF || s.Type == obj.SHOSTOBJ {
				s.Dynimplib = lib
				s.Extname = remote
				s.Dynimpvers = q
				if s.Type != obj.SHOSTOBJ {
					s.Type = obj.SDYNIMPORT
				}
				havedynamic = 1
			}

			continue
		}

		if f[0] == "cgo_import_static" {
			if len(f) != 2 {
				goto err
			}
			local = f[1]
			s = Linklookup(Ctxt, local, 0)
			s.Type = obj.SHOSTOBJ
			s.Size = 0
			continue
		}

		if f[0] == "cgo_export_static" || f[0] == "cgo_export_dynamic" {
			if len(f) < 2 || len(f) > 3 {
				goto err
			}
			local = f[1]
			if len(f) > 2 {
				remote = f[2]
			} else {
				remote = local
			}
			local = expandpkg(local, pkg)
			s = Linklookup(Ctxt, local, 0)

			switch Buildmode {
			case BuildmodeCShared, BuildmodeCArchive:
				if s == Linklookup(Ctxt, "main", 0) {
					continue
				}
			}

			// export overrides import, for openbsd/cgo.
			// see issue 4878.
			if s.Dynimplib != "" {
				s.Dynimplib = ""
				s.Extname = ""
				s.Dynimpvers = ""
				s.Type = 0
			}

			if s.Cgoexport == 0 {
				s.Extname = remote
				dynexp = append(dynexp, s)
			} else if s.Extname != remote {
				fmt.Fprintf(os.Stderr, "%s: conflicting cgo_export directives: %s as %s and %s\n", os.Args[0], s.Name, s.Extname, remote)
				nerrors++
				return
			}

			if f[0] == "cgo_export_static" {
				s.Cgoexport |= CgoExportStatic
			} else {
				s.Cgoexport |= CgoExportDynamic
			}
			if local != f[1] {
			}
			continue
		}

		if f[0] == "cgo_dynamic_linker" {
			if len(f) != 2 {
				goto err
			}

			if Debug['I'] == 0 {
				if interpreter != "" && interpreter != f[1] {
					fmt.Fprintf(os.Stderr, "%s: conflict dynlinker: %s and %s\n", os.Args[0], interpreter, f[1])
					nerrors++
					return
				}

				interpreter = f[1]
			}

			continue
		}

		if f[0] == "cgo_ldflag" {
			if len(f) != 2 {
				goto err
			}
			ldflag = append(ldflag, f[1])
			continue
		}
	}

	return

err:
	fmt.Fprintf(os.Stderr, "%s: %s: invalid dynimport line: %s\n", os.Args[0], file, p0)
	nerrors++
}

var seenlib = make(map[string]bool)

func adddynlib(lib string) {
	if seenlib[lib] || Linkmode == LinkExternal {
		return
	}
	seenlib[lib] = true

	if Iself {
		s := Linklookup(Ctxt, ".dynstr", 0)
		if s.Size == 0 {
			Addstring(s, "")
		}
		Elfwritedynent(Linklookup(Ctxt, ".dynamic", 0), DT_NEEDED, uint64(Addstring(s, lib)))
	} else {
		Diag("adddynlib: unsupported binary format")
	}
}

func Adddynsym(ctxt *Link, s *LSym) {
	if s.Dynid >= 0 || Linkmode == LinkExternal {
		return
	}

	if Iself {
		Elfadddynsym(ctxt, s)
	} else if HEADTYPE == obj.Hdarwin {
		Diag("adddynsym: missed symbol %s (%s)", s.Name, s.Extname)
	} else if HEADTYPE == obj.Hwindows {
		// already taken care of
	} else {
		Diag("adddynsym: unsupported binary format")
	}
}

var markq *LSym

var emarkq *LSym

func mark1(s *LSym, parent *LSym) {
	if s == nil || s.Reachable {
		return
	}
	if strings.HasPrefix(s.Name, "go.weak.") {
		return
	}
	s.Reachable = true
	s.Reachparent = parent
	if markq == nil {
		markq = s
	} else {
		emarkq.Queue = s
	}
	emarkq = s
}

func mark(s *LSym) {
	mark1(s, nil)
}

func markflood() {
	var a *Auto
	var i int

	for s := markq; s != nil; s = s.Queue {
		if s.Type == obj.STEXT {
			if Debug['v'] > 1 {
				fmt.Fprintf(&Bso, "marktext %s\n", s.Name)
			}
			for a = s.Autom; a != nil; a = a.Link {
				mark1(a.Gotype, s)
			}
		}

		for i = 0; i < len(s.R); i++ {
			mark1(s.R[i].Sym, s)
		}
		if s.Pcln != nil {
			for i = 0; i < s.Pcln.Nfuncdata; i++ {
				mark1(s.Pcln.Funcdata[i], s)
			}
		}

		mark1(s.Gotype, s)
		mark1(s.Sub, s)
		mark1(s.Outer, s)
	}
}

var markextra = []string{
	"runtime.morestack",
	"runtime.morestackx",
	"runtime.morestack00",
	"runtime.morestack10",
	"runtime.morestack01",
	"runtime.morestack11",
	"runtime.morestack8",
	"runtime.morestack16",
	"runtime.morestack24",
	"runtime.morestack32",
	"runtime.morestack40",
	"runtime.morestack48",
	// on arm, lock in the div/mod helpers too
	"_div",
	"_divu",
	"_mod",
	"_modu",
}

func deadcode() {
	if Debug['v'] != 0 {
		fmt.Fprintf(&Bso, "%5.2f deadcode\n", obj.Cputime())
	}

	if Buildmode == BuildmodeShared {
		// Mark all symbols defined in this library as reachable when
		// building a shared library.
		for s := Ctxt.Allsym; s != nil; s = s.Allsym {
			if s.Type != 0 && s.Type != obj.SDYNIMPORT {
				mark(s)
			}
		}
		markflood()
	} else {
		mark(Linklookup(Ctxt, INITENTRY, 0))
		if Linkshared && Buildmode == BuildmodeExe {
			mark(Linkrlookup(Ctxt, "main.main", 0))
			mark(Linkrlookup(Ctxt, "main.init", 0))
		}
		for i := 0; i < len(markextra); i++ {
			mark(Linklookup(Ctxt, markextra[i], 0))
		}

		for i := 0; i < len(dynexp); i++ {
			mark(dynexp[i])
		}
		markflood()

		// keep each beginning with 'typelink.' if the symbol it points at is being kept.
		for s := Ctxt.Allsym; s != nil; s = s.Allsym {
			if strings.HasPrefix(s.Name, "go.typelink.") {
				s.Reachable = len(s.R) == 1 && s.R[0].Sym.Reachable
			}
		}

		// remove dead text but keep file information (z symbols).
		var last *LSym

		for s := Ctxt.Textp; s != nil; s = s.Next {
			if !s.Reachable {
				continue
			}

			// NOTE: Removing s from old textp and adding to new, shorter textp.
			if last == nil {
				Ctxt.Textp = s
			} else {
				last.Next = s
			}
			last = s
		}

		if last == nil {
			Ctxt.Textp = nil
			Ctxt.Etextp = nil
		} else {
			last.Next = nil
			Ctxt.Etextp = last
		}
	}

	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if strings.HasPrefix(s.Name, "go.weak.") {
			s.Special = 1 // do not lay out in data segment
			s.Reachable = true
			s.Hide = 1
		}
	}

	// record field tracking references
	var buf bytes.Buffer
	var p *LSym
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if strings.HasPrefix(s.Name, "go.track.") {
			s.Special = 1 // do not lay out in data segment
			s.Hide = 1
			if s.Reachable {
				buf.WriteString(s.Name[9:])
				for p = s.Reachparent; p != nil; p = p.Reachparent {
					buf.WriteString("\t")
					buf.WriteString(p.Name)
				}
				buf.WriteString("\n")
			}

			s.Type = obj.SCONST
			s.Value = 0
		}
	}

	if tracksym == "" {
		return
	}
	s := Linklookup(Ctxt, tracksym, 0)
	if !s.Reachable {
		return
	}
	addstrdata(tracksym, buf.String())
}

func doweak() {
	var t *LSym

	// resolve weak references only if
	// target symbol will be in binary anyway.
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if strings.HasPrefix(s.Name, "go.weak.") {
			t = Linkrlookup(Ctxt, s.Name[8:], int(s.Version))
			if t != nil && t.Type != 0 && t.Reachable {
				s.Value = t.Value
				s.Type = t.Type
				s.Outer = t
			} else {
				s.Type = obj.SCONST
				s.Value = 0
			}

			continue
		}
	}
}

func addexport() {
	if HEADTYPE == obj.Hdarwin {
		return
	}

	for _, exp := range dynexp {
		Adddynsym(Ctxt, exp)
	}
	for _, lib := range dynlib {
		adddynlib(lib)
	}
}

type Pkg struct {
	mark    bool
	checked bool
	path    string
	impby   []*Pkg
}

var (
	// pkgmap records the imported-by relationship between packages.
	// Entries are keyed by package path (e.g., "runtime" or "net/url").
	pkgmap = map[string]*Pkg{}

	pkgall []*Pkg
)

func lookupPkg(path string) *Pkg {
	if p, ok := pkgmap[path]; ok {
		return p
	}
	p := &Pkg{path: path}
	pkgmap[path] = p
	pkgall = append(pkgall, p)
	return p
}

// imported records that package pkg imports package imp.
func imported(pkg, imp string) {
	// everyone imports runtime, even runtime.
	if imp == "runtime" {
		return
	}

	p := lookupPkg(pkg)
	i := lookupPkg(imp)
	i.impby = append(i.impby, p)
}

func (p *Pkg) cycle() *Pkg {
	if p.checked {
		return nil
	}

	if p.mark {
		nerrors++
		fmt.Printf("import cycle:\n")
		fmt.Printf("\t%s\n", p.path)
		return p
	}

	p.mark = true
	for _, q := range p.impby {
		if bad := q.cycle(); bad != nil {
			p.mark = false
			p.checked = true
			fmt.Printf("\timports %s\n", p.path)
			if bad == p {
				return nil
			}
			return bad
		}
	}

	p.checked = true
	p.mark = false
	return nil
}

func importcycles() {
	for _, p := range pkgall {
		p.cycle()
	}
}

func setlinkmode(arg string) {
	if arg == "internal" {
		Linkmode = LinkInternal
	} else if arg == "external" {
		Linkmode = LinkExternal
	} else if arg == "auto" {
		Linkmode = LinkAuto
	} else {
		Exitf("unknown link mode -linkmode %s", arg)
	}
}
