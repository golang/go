// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

package ld

import (
	"cmd/internal/bio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

// TODO:
//	generate debugging section in binary.
//	once the dust settles, try to move some code to
//		libmach, so that other linkers and ar can share.

func ldpkg(ctxt *Link, f *bio.Reader, lib *sym.Library, length int64, filename string) {
	if *flagG {
		return
	}

	if int64(int(length)) != length {
		fmt.Fprintf(os.Stderr, "%s: too much pkg data in %s\n", os.Args[0], filename)
		return
	}

	bdata := make([]byte, length)
	if _, err := io.ReadFull(f, bdata); err != nil {
		fmt.Fprintf(os.Stderr, "%s: short pkg read %s\n", os.Args[0], filename)
		return
	}
	data := string(bdata)

	// process header lines
	for data != "" {
		var line string
		line, data, _ = strings.Cut(data, "\n")
		if line == "main" {
			lib.Main = true
		}
		if line == "" {
			break
		}
	}

	// look for cgo section
	p0 := strings.Index(data, "\n$$  // cgo")
	var p1 int
	if p0 >= 0 {
		p0 += p1
		i := strings.IndexByte(data[p0+1:], '\n')
		if i < 0 {
			fmt.Fprintf(os.Stderr, "%s: found $$ // cgo but no newline in %s\n", os.Args[0], filename)
			return
		}
		p0 += 1 + i

		p1 = strings.Index(data[p0:], "\n$$")
		if p1 < 0 {
			p1 = strings.Index(data[p0:], "\n!\n")
		}
		if p1 < 0 {
			fmt.Fprintf(os.Stderr, "%s: cannot find end of // cgo section in %s\n", os.Args[0], filename)
			return
		}
		p1 += p0
		loadcgo(ctxt, filename, objabi.PathToPrefix(lib.Pkg), data[p0:p1])
	}
}

func loadcgo(ctxt *Link, file string, pkg string, p string) {
	var directives [][]string
	if err := json.NewDecoder(strings.NewReader(p)).Decode(&directives); err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s: failed decoding cgo directives: %v\n", os.Args[0], file, err)
		nerrors++
		return
	}

	// Record the directives. We'll process them later after Symbols are created.
	ctxt.cgodata = append(ctxt.cgodata, cgodata{file, pkg, directives})
}

// Set symbol attributes or flags based on cgo directives.
// Any newly discovered HOSTOBJ syms are added to 'hostObjSyms'.
func setCgoAttr(ctxt *Link, file string, pkg string, directives [][]string, hostObjSyms map[loader.Sym]struct{}) {
	l := ctxt.loader
	for _, f := range directives {
		switch f[0] {
		case "cgo_import_dynamic":
			if len(f) < 2 || len(f) > 4 {
				break
			}

			local := f[1]
			remote := local
			if len(f) > 2 {
				remote = f[2]
			}
			lib := ""
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

				if ctxt.HeadType == objabi.Hdarwin {
					machoadddynlib(lib, ctxt.LinkMode)
				} else {
					dynlib = append(dynlib, lib)
				}
				continue
			}

			q := ""
			if before, after, found := strings.Cut(remote, "#"); found {
				remote, q = before, after
			}
			s := l.LookupOrCreateSym(local, 0)
			st := l.SymType(s)
			if st == 0 || st == sym.SXREF || st == sym.SBSS || st == sym.SNOPTRBSS || st == sym.SHOSTOBJ {
				l.SetSymDynimplib(s, lib)
				l.SetSymExtname(s, remote)
				l.SetSymDynimpvers(s, q)
				if st != sym.SHOSTOBJ {
					su := l.MakeSymbolUpdater(s)
					su.SetType(sym.SDYNIMPORT)
				} else {
					hostObjSyms[s] = struct{}{}
				}
				havedynamic = 1
				if lib != "" && ctxt.IsDarwin() {
					machoadddynlib(lib, ctxt.LinkMode)
				}
			}

			continue

		case "cgo_import_static":
			if len(f) != 2 {
				break
			}
			local := f[1]

			s := l.LookupOrCreateSym(local, 0)
			su := l.MakeSymbolUpdater(s)
			su.SetType(sym.SHOSTOBJ)
			su.SetSize(0)
			hostObjSyms[s] = struct{}{}
			continue

		case "cgo_export_static", "cgo_export_dynamic":
			if len(f) < 2 || len(f) > 4 {
				break
			}
			local := f[1]
			remote := local
			if len(f) > 2 {
				remote = f[2]
			}
			// The compiler adds a fourth argument giving
			// the definition ABI of function symbols.
			abi := obj.ABI0
			if len(f) > 3 {
				var ok bool
				abi, ok = obj.ParseABI(f[3])
				if !ok {
					fmt.Fprintf(os.Stderr, "%s: bad ABI in cgo_export directive %s\n", os.Args[0], f)
					nerrors++
					return
				}
			}

			s := l.LookupOrCreateSym(local, sym.ABIToVersion(abi))

			if l.SymType(s) == sym.SHOSTOBJ {
				hostObjSyms[s] = struct{}{}
			}

			switch ctxt.BuildMode {
			case BuildModeCShared, BuildModeCArchive, BuildModePlugin:
				if s == l.Lookup("main", 0) {
					continue
				}
			}

			// export overrides import, for openbsd/cgo.
			// see issue 4878.
			if l.SymDynimplib(s) != "" {
				l.SetSymDynimplib(s, "")
				l.SetSymDynimpvers(s, "")
				l.SetSymExtname(s, "")
				var su *loader.SymbolBuilder
				su = l.MakeSymbolUpdater(s)
				su.SetType(0)
			}

			if !(l.AttrCgoExportStatic(s) || l.AttrCgoExportDynamic(s)) {
				l.SetSymExtname(s, remote)
			} else if l.SymExtname(s) != remote {
				fmt.Fprintf(os.Stderr, "%s: conflicting cgo_export directives: %s as %s and %s\n", os.Args[0], l.SymName(s), l.SymExtname(s), remote)
				nerrors++
				return
			}

			// Mark exported symbols and also add them to
			// the lists used for roots in the deadcode pass.
			if f[0] == "cgo_export_static" {
				if ctxt.LinkMode == LinkExternal && !l.AttrCgoExportStatic(s) {
					// Static cgo exports appear
					// in the exported symbol table.
					ctxt.dynexp = append(ctxt.dynexp, s)
				}
				if ctxt.LinkMode == LinkInternal {
					// For internal linking, we're
					// responsible for resolving
					// relocations from host objects.
					// Record the right Go symbol
					// version to use.
					l.AddCgoExport(s)
				}
				l.SetAttrCgoExportStatic(s, true)
			} else {
				if ctxt.LinkMode == LinkInternal && !l.AttrCgoExportDynamic(s) {
					// Dynamic cgo exports appear
					// in the exported symbol table.
					ctxt.dynexp = append(ctxt.dynexp, s)
				}
				l.SetAttrCgoExportDynamic(s, true)
			}

			continue

		case "cgo_dynamic_linker":
			if len(f) != 2 {
				break
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

		case "cgo_ldflag":
			if len(f) != 2 {
				break
			}
			ldflag = append(ldflag, f[1])
			continue
		}

		fmt.Fprintf(os.Stderr, "%s: %s: invalid cgo directive: %q\n", os.Args[0], file, f)
		nerrors++
	}
	return
}

// openbsdTrimLibVersion indicates whether a shared library is
// versioned and if it is, returns the unversioned name. The
// OpenBSD library naming scheme is lib<name>.so.<major>.<minor>
func openbsdTrimLibVersion(lib string) (string, bool) {
	parts := strings.Split(lib, ".")
	if len(parts) != 4 {
		return "", false
	}
	if parts[1] != "so" {
		return "", false
	}
	if _, err := strconv.Atoi(parts[2]); err != nil {
		return "", false
	}
	if _, err := strconv.Atoi(parts[3]); err != nil {
		return "", false
	}
	return fmt.Sprintf("%s.%s", parts[0], parts[1]), true
}

// dedupLibrariesOpenBSD dedups a list of shared libraries, treating versioned
// and unversioned libraries as equivalents. Versioned libraries are preferred
// and retained over unversioned libraries. This avoids the situation where
// the use of cgo results in a DT_NEEDED for a versioned library (for example,
// libc.so.96.1), while a dynamic import specifies an unversioned library (for
// example, libc.so) - this would otherwise result in two DT_NEEDED entries
// for the same library, resulting in a failure when ld.so attempts to load
// the Go binary.
func dedupLibrariesOpenBSD(ctxt *Link, libs []string) []string {
	libraries := make(map[string]string)
	for _, lib := range libs {
		if name, ok := openbsdTrimLibVersion(lib); ok {
			// Record unversioned name as seen.
			seenlib[name] = true
			libraries[name] = lib
		} else if _, ok := libraries[lib]; !ok {
			libraries[lib] = lib
		}
	}

	libs = nil
	for _, lib := range libraries {
		libs = append(libs, lib)
	}
	sort.Strings(libs)

	return libs
}

func dedupLibraries(ctxt *Link, libs []string) []string {
	if ctxt.Target.IsOpenbsd() {
		return dedupLibrariesOpenBSD(ctxt, libs)
	}
	return libs
}

var seenlib = make(map[string]bool)

func adddynlib(ctxt *Link, lib string) {
	if seenlib[lib] || ctxt.LinkMode == LinkExternal {
		return
	}
	seenlib[lib] = true

	if ctxt.IsELF {
		dsu := ctxt.loader.MakeSymbolUpdater(ctxt.DynStr)
		if dsu.Size() == 0 {
			dsu.Addstring("")
		}
		du := ctxt.loader.MakeSymbolUpdater(ctxt.Dynamic)
		Elfwritedynent(ctxt.Arch, du, elf.DT_NEEDED, uint64(dsu.Addstring(lib)))
	} else {
		Errorf(nil, "adddynlib: unsupported binary format")
	}
}

func Adddynsym(ldr *loader.Loader, target *Target, syms *ArchSyms, s loader.Sym) {
	if ldr.SymDynid(s) >= 0 || target.LinkMode == LinkExternal {
		return
	}

	if target.IsELF {
		elfadddynsym(ldr, target, syms, s)
	} else if target.HeadType == objabi.Hdarwin {
		ldr.Errorf(s, "adddynsym: missed symbol (Extname=%s)", ldr.SymExtname(s))
	} else if target.HeadType == objabi.Hwindows {
		// already taken care of
	} else {
		ldr.Errorf(s, "adddynsym: unsupported binary format")
	}
}

func fieldtrack(arch *sys.Arch, l *loader.Loader) {
	var buf strings.Builder
	for i := loader.Sym(1); i < loader.Sym(l.NSym()); i++ {
		if name := l.SymName(i); strings.HasPrefix(name, "go:track.") {
			if l.AttrReachable(i) {
				l.SetAttrSpecial(i, true)
				l.SetAttrNotInSymbolTable(i, true)
				buf.WriteString(name[9:])
				for p := l.Reachparent[i]; p != 0; p = l.Reachparent[p] {
					buf.WriteString("\t")
					buf.WriteString(l.SymName(p))
				}
				buf.WriteString("\n")
			}
		}
	}
	l.Reachparent = nil // we are done with it
	if *flagFieldTrack == "" {
		return
	}
	s := l.Lookup(*flagFieldTrack, 0)
	if s == 0 || !l.AttrReachable(s) {
		return
	}
	bld := l.MakeSymbolUpdater(s)
	bld.SetType(sym.SDATA)
	addstrdata(arch, l, *flagFieldTrack, buf.String())
}

func (ctxt *Link) addexport() {
	// Track undefined external symbols during external link.
	if ctxt.LinkMode == LinkExternal {
		for _, s := range ctxt.Textp {
			if ctxt.loader.AttrSpecial(s) || ctxt.loader.AttrSubSymbol(s) {
				continue
			}
			relocs := ctxt.loader.Relocs(s)
			for i := 0; i < relocs.Count(); i++ {
				if rs := relocs.At(i).Sym(); rs != 0 {
					if ctxt.loader.SymType(rs) == sym.Sxxx && !ctxt.loader.AttrLocal(rs) {
						// sanity check
						if len(ctxt.loader.Data(rs)) != 0 {
							panic("expected no data on undef symbol")
						}
						su := ctxt.loader.MakeSymbolUpdater(rs)
						su.SetType(sym.SUNDEFEXT)
					}
				}
			}
		}
	}

	// TODO(aix)
	if ctxt.HeadType == objabi.Hdarwin || ctxt.HeadType == objabi.Haix {
		return
	}

	// Add dynamic symbols.
	for _, s := range ctxt.dynexp {
		// Consistency check.
		if !ctxt.loader.AttrReachable(s) {
			panic("dynexp entry not reachable")
		}

		Adddynsym(ctxt.loader, &ctxt.Target, &ctxt.ArchSyms, s)
	}

	for _, lib := range dedupLibraries(ctxt, dynlib) {
		adddynlib(ctxt, lib)
	}
}
