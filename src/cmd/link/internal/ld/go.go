// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

package ld

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/obj"
	"fmt"
	"io"
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

func ldpkg(ctxt *Link, f *bio.Reader, pkg string, length int64, filename string, whence int) {
	var p0, p1 int

	if *flagG {
		return
	}

	if int64(int(length)) != length {
		fmt.Fprintf(os.Stderr, "%s: too much pkg data in %s\n", os.Args[0], filename)
		if *flagU {
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
	if _, err := io.ReadFull(f, bdata); err != nil {
		fmt.Fprintf(os.Stderr, "%s: short pkg read %s\n", os.Args[0], filename)
		if *flagU {
			errorexit()
		}
		return
	}
	data := string(bdata)

	// process header lines
	isSafe := false
	isMain := false
	for data != "" {
		var line string
		if i := strings.Index(data, "\n"); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			line, data = data, ""
		}
		if line == "safe" {
			isSafe = true
		}
		if line == "main" {
			isMain = true
		}
		if line == "" {
			break
		}
	}

	if whence == Pkgdef || whence == FileObj {
		if pkg == "main" && !isMain {
			Exitf("%s: not package main", filename)
		}
		if *flagU && whence != ArchiveObj && !isSafe {
			Exitf("load of unsafe package %s", filename)
		}
	}

	// __.PKGDEF has no cgo section - those are in the C compiler-generated object files.
	if whence == Pkgdef {
		return
	}

	// look for cgo section
	p0 = strings.Index(data, "\n$$  // cgo")
	if p0 >= 0 {
		p0 += p1
		i := strings.IndexByte(data[p0+1:], '\n')
		if i < 0 {
			fmt.Fprintf(os.Stderr, "%s: found $$ // cgo but no newline in %s\n", os.Args[0], filename)
			if *flagU {
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
			if *flagU {
				errorexit()
			}
			return
		}
		p1 += p0

		loadcgo(ctxt, filename, pkg, data[p0:p1])
	}
}

func loadcgo(ctxt *Link, file string, pkg string, p string) {
	var next string
	var q string
	var f []string
	var local string
	var remote string
	var lib string
	var s *Symbol

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

			if *FlagD {
				fmt.Fprintf(os.Stderr, "%s: %s: cannot use dynamic imports with -d flag\n", os.Args[0], file)
				nerrors++
				return
			}

			if local == "_" && remote == "_" {
				// allow #pragma dynimport _ _ "foo.so"
				// to force a link of foo.so.
				havedynamic = 1

				if Headtype == obj.Hdarwin {
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
			s = ctxt.Syms.Lookup(local, 0)
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
			s = ctxt.Syms.Lookup(local, 0)
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
			s = ctxt.Syms.Lookup(local, 0)

			switch Buildmode {
			case BuildmodeCShared, BuildmodeCArchive, BuildmodePlugin:
				if s == ctxt.Syms.Lookup("main", 0) {
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

			if !s.Attr.CgoExport() {
				s.Extname = remote
				dynexp = append(dynexp, s)
			} else if s.Extname != remote {
				fmt.Fprintf(os.Stderr, "%s: conflicting cgo_export directives: %s as %s and %s\n", os.Args[0], s.Name, s.Extname, remote)
				nerrors++
				return
			}

			if f[0] == "cgo_export_static" {
				s.Attr |= AttrCgoExportStatic
			} else {
				s.Attr |= AttrCgoExportDynamic
			}
			if local != f[1] {
			}
			continue
		}

		if f[0] == "cgo_dynamic_linker" {
			if len(f) != 2 {
				goto err
			}

			if *flagInterpreter == "" {
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

func adddynlib(ctxt *Link, lib string) {
	if seenlib[lib] || Linkmode == LinkExternal {
		return
	}
	seenlib[lib] = true

	if Iself {
		s := ctxt.Syms.Lookup(".dynstr", 0)
		if s.Size == 0 {
			Addstring(s, "")
		}
		Elfwritedynent(ctxt, ctxt.Syms.Lookup(".dynamic", 0), DT_NEEDED, uint64(Addstring(s, lib)))
	} else {
		Errorf(nil, "adddynlib: unsupported binary format")
	}
}

func Adddynsym(ctxt *Link, s *Symbol) {
	if s.Dynid >= 0 || Linkmode == LinkExternal {
		return
	}

	if Iself {
		Elfadddynsym(ctxt, s)
	} else if Headtype == obj.Hdarwin {
		Errorf(s, "adddynsym: missed symbol (Extname=%s)", s.Extname)
	} else if Headtype == obj.Hwindows {
		// already taken care of
	} else {
		Errorf(s, "adddynsym: unsupported binary format")
	}
}

func fieldtrack(ctxt *Link) {
	// record field tracking references
	var buf bytes.Buffer
	for _, s := range ctxt.Syms.Allsym {
		if strings.HasPrefix(s.Name, "go.track.") {
			s.Attr |= AttrSpecial // do not lay out in data segment
			s.Attr |= AttrHidden
			if s.Attr.Reachable() {
				buf.WriteString(s.Name[9:])
				for p := s.Reachparent; p != nil; p = p.Reachparent {
					buf.WriteString("\t")
					buf.WriteString(p.Name)
				}
				buf.WriteString("\n")
			}

			s.Type = obj.SCONST
			s.Value = 0
		}
	}

	if *flagFieldTrack == "" {
		return
	}
	s := ctxt.Syms.Lookup(*flagFieldTrack, 0)
	if !s.Attr.Reachable() {
		return
	}
	addstrdata(ctxt, *flagFieldTrack, buf.String())
}

func (ctxt *Link) addexport() {
	if Headtype == obj.Hdarwin {
		return
	}

	for _, exp := range dynexp {
		Adddynsym(ctxt, exp)
	}
	for _, lib := range dynlib {
		adddynlib(ctxt, lib)
	}
}

type Pkg struct {
	mark    bool
	checked bool
	path    string
	impby   []*Pkg
}

var pkgall []*Pkg

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
