// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

// replace all "". with pkg.
func expandpkg(t0 string, pkg string) string {
	return strings.Replace(t0, `"".`, pkg+".", -1)
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

// accumulate all type information from .6 files.
// check for inconsistencies.

// TODO:
//	generate debugging section in binary.
//	once the dust settles, try to move some code to
//		libmach, so that other linkers and ar can share.

/*
 *	package import data
 */
type Import struct {
	prefix string // "type", "var", "func", "const"
	name   string
	def    string
	file   string
}

// importmap records type information about imported symbols to detect inconsistencies.
// Entries are keyed by qualified symbol name (e.g., "runtime.Callers" or "net/url.Error").
var importmap = map[string]*Import{}

func lookupImport(name string) *Import {
	if x, ok := importmap[name]; ok {
		return x
	}
	x := &Import{name: name}
	importmap[name] = x
	return x
}

func ldpkg(f *Biobuf, pkg string, length int64, filename string, whence int) {
	var p0, p1 int

	if Debug['g'] != 0 {
		return
	}

	if int64(int(length)) != length {
		fmt.Fprintf(os.Stderr, "%s: too much pkg data in %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			Errorexit()
		}
		return
	}

	bdata := make([]byte, length)
	if int64(Bread(f, bdata)) != length {
		fmt.Fprintf(os.Stderr, "%s: short pkg read %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			Errorexit()
		}
		return
	}
	data := string(bdata)

	// first \n$$ marks beginning of exports - skip rest of line
	p0 = strings.Index(data, "\n$$")
	if p0 < 0 {
		if Debug['u'] != 0 && whence != ArchiveObj {
			fmt.Fprintf(os.Stderr, "%s: cannot find export data in %s\n", os.Args[0], filename)
			Errorexit()
		}
		return
	}

	p0 += 3
	for p0 < len(data) && data[p0] != '\n' {
		p0++
	}

	// second marks end of exports / beginning of local data
	p1 = strings.Index(data[p0:], "\n$$")
	if p1 < 0 {
		fmt.Fprintf(os.Stderr, "%s: cannot find end of exports in %s\n", os.Args[0], filename)
		if Debug['u'] != 0 {
			Errorexit()
		}
		return
	}
	p1 += p0

	for p0 < p1 && (data[p0] == ' ' || data[p0] == '\t' || data[p0] == '\n') {
		p0++
	}
	if p0 < p1 {
		if !strings.HasPrefix(data[p0:], "package ") {
			fmt.Fprintf(os.Stderr, "%s: bad package section in %s - %.20s\n", os.Args[0], filename, data[p0:])
			if Debug['u'] != 0 {
				Errorexit()
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
			fmt.Fprintf(os.Stderr, "%s: load of unsafe package %s\n", os.Args[0], filename)
			nerrors++
			Errorexit()
		}

		name := data[pname:p0]
		for p0 < p1 && data[p0] != '\n' {
			p0++
		}
		if p0 < p1 {
			p0++
		}

		if pkg == "main" && name != "main" {
			fmt.Fprintf(os.Stderr, "%s: %s: not package main (package %s)\n", os.Args[0], filename, name)
			nerrors++
			Errorexit()
		}

		loadpkgdata(filename, pkg, data[p0:p1])
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
				Errorexit()
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
				Errorexit()
			}
			return
		}
		p1 += p0

		loadcgo(filename, pkg, data[p0:p1])
	}
}

func loadpkgdata(file string, pkg string, data string) {
	var prefix string
	var name string
	var def string

	p := data
	for parsepkgdata(file, pkg, &p, &prefix, &name, &def) > 0 {
		x := lookupImport(name)
		if x.prefix == "" {
			x.prefix = prefix
			x.def = def
			x.file = file
		} else if x.prefix != prefix {
			fmt.Fprintf(os.Stderr, "%s: conflicting definitions for %s\n", os.Args[0], name)
			fmt.Fprintf(os.Stderr, "%s:\t%s %s ...\n", x.file, x.prefix, name)
			fmt.Fprintf(os.Stderr, "%s:\t%s %s ...\n", file, prefix, name)
			nerrors++
		} else if x.def != def {
			fmt.Fprintf(os.Stderr, "%s: conflicting definitions for %s\n", os.Args[0], name)
			fmt.Fprintf(os.Stderr, "%s:\t%s %s %s\n", x.file, x.prefix, name, x.def)
			fmt.Fprintf(os.Stderr, "%s:\t%s %s %s\n", file, prefix, name, def)
			nerrors++
		}
	}
}

func parsepkgdata(file string, pkg string, pp *string, prefixp *string, namep *string, defp *string) int {
	// skip white space
	p := *pp

loop:
	for len(p) > 0 && (p[0] == ' ' || p[0] == '\t' || p[0] == '\n') {
		p = p[1:]
	}
	if len(p) == 0 || strings.HasPrefix(p, "$$\n") {
		return 0
	}

	// prefix: (var|type|func|const)
	prefix := p

	if len(p) < 7 {
		return -1
	}
	if strings.HasPrefix(p, "var ") {
		p = p[4:]
	} else if strings.HasPrefix(p, "type ") {
		p = p[5:]
	} else if strings.HasPrefix(p, "func ") {
		p = p[5:]
	} else if strings.HasPrefix(p, "const ") {
		p = p[6:]
	} else if strings.HasPrefix(p, "import ") {
		p = p[7:]
		for len(p) > 0 && p[0] != ' ' {
			p = p[1:]
		}
		p = p[1:]
		line := p
		for len(p) > 0 && p[0] != '\n' {
			p = p[1:]
		}
		if len(p) == 0 {
			fmt.Fprintf(os.Stderr, "%s: %s: confused in import line\n", os.Args[0], file)
			nerrors++
			return -1
		}
		line = line[:len(line)-len(p)]
		line = strings.TrimSuffix(line, " // indirect")
		path, err := strconv.Unquote(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %s: confused in import path: %q\n", os.Args[0], file, line)
			nerrors++
			return -1
		}
		p = p[1:]
		imported(pkg, path)
		goto loop
	} else {
		fmt.Fprintf(os.Stderr, "%s: %s: confused in pkg data near <<%.40s>>\n", os.Args[0], file, prefix)
		nerrors++
		return -1
	}

	prefix = prefix[:len(prefix)-len(p)-1]

	// name: a.b followed by space
	name := p

	inquote := false
	for len(p) > 0 {
		if p[0] == ' ' && !inquote {
			break
		}

		if p[0] == '\\' {
			p = p[1:]
		} else if p[0] == '"' {
			inquote = !inquote
		}

		p = p[1:]
	}

	if len(p) == 0 {
		return -1
	}
	name = name[:len(name)-len(p)]
	p = p[1:]

	// def: free form to new line
	def := p

	for len(p) > 0 && p[0] != '\n' {
		p = p[1:]
	}
	if len(p) == 0 {
		return -1
	}
	def = def[:len(def)-len(p)]
	var defbuf *bytes.Buffer
	p = p[1:]

	// include methods on successive lines in def of named type
	var meth string
	for parsemethod(&p, &meth) > 0 {
		if defbuf == nil {
			defbuf = new(bytes.Buffer)
			defbuf.WriteString(def)
		}
		defbuf.WriteString("\n\t")
		defbuf.WriteString(meth)
	}
	if defbuf != nil {
		def = defbuf.String()
	}

	name = expandpkg(name, pkg)
	def = expandpkg(def, pkg)

	// done
	*pp = p

	*prefixp = prefix
	*namep = name
	*defp = def
	return 1
}

func parsemethod(pp *string, methp *string) int {
	// skip white space
	p := *pp

	for len(p) > 0 && (p[0] == ' ' || p[0] == '\t') {
		p = p[1:]
	}
	if len(p) == 0 {
		return 0
	}

	// might be a comment about the method
	if strings.HasPrefix(p, "//") {
		goto useline
	}

	// if it says "func (", it's a method
	if strings.HasPrefix(p, "func (") {
		goto useline
	}
	return 0

	// definition to end of line
useline:
	*methp = p

	for len(p) > 0 && p[0] != '\n' {
		p = p[1:]
	}
	if len(p) == 0 {
		fmt.Fprintf(os.Stderr, "%s: lost end of line in method definition\n", os.Args[0])
		*pp = ""
		return -1
	}

	*methp = (*methp)[:len(*methp)-len(p)]
	*pp = p[1:]
	return 1
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

				Thearch.Adddynlib(lib)
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
			if s.Type == 0 || s.Type == SXREF || s.Type == SHOSTOBJ {
				s.Dynimplib = lib
				s.Extname = remote
				s.Dynimpvers = q
				if s.Type != SHOSTOBJ {
					s.Type = SDYNIMPORT
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
			s.Type = SHOSTOBJ
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
		if s.Type == STEXT {
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

	if Buildmode == BuildmodeShared || Buildmode == BuildmodeCArchive {
		// Mark all symbols as reachable when building a
		// shared library.
		for s := Ctxt.Allsym; s != nil; s = s.Allsym {
			if s.Type != 0 {
				mark(s)
			}
		}
		mark(Linkrlookup(Ctxt, "main.main", 0))
		mark(Linkrlookup(Ctxt, "main.init", 0))
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

			s.Type = SCONST
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
				s.Type = SCONST
				s.Value = 0
			}

			continue
		}
	}
}

func addexport() {
	if HEADTYPE == Hdarwin {
		return
	}

	for i := 0; i < len(dynexp); i++ {
		Thearch.Adddynsym(Ctxt, dynexp[i])
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
		fmt.Fprintf(os.Stderr, "unknown link mode -linkmode %s\n", arg)
		Errorexit()
	}
}
