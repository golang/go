// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Initial scan of packages making up a program.

// TODO(rsc): Rename goobj.SymID.Version to StaticID to avoid confusion with the ELF meaning of version.
// TODO(rsc): Fix file format so that SBSS/SNOPTRBSS with data is listed as SDATA/SNOPTRDATA.
// TODO(rsc): Parallelize scan to overlap file i/o where possible.

package main

import (
	"debug/goobj"
	"os"
	"strings"
)

// scan scans all packages making up the program, starting with package main defined in mainfile.
func (p *Prog) scan(mainfile string) {
	p.initScan()
	p.scanFile("main", mainfile)
	if len(p.Missing) != 0 {
		// TODO(rsc): iterate in deterministic order
		for sym := range p.Missing {
			p.errorf("undefined: %s", sym)
		}
	}

	// TODO(rsc): Walk import graph to diagnose cycles.
}

// initScan initializes the Prog fields needed by scan.
func (p *Prog) initScan() {
	p.Packages = make(map[string]*Package)
	p.Syms = make(map[goobj.SymID]*Sym)
	p.Missing = make(map[goobj.SymID]bool)
	p.Missing[startSymID] = true
}

// scanFile reads file to learn about the package with the given import path.
func (p *Prog) scanFile(pkgpath string, file string) {
	pkg := &Package{
		File: file,
	}
	p.Packages[pkgpath] = pkg

	f, err := os.Open(file)
	if err != nil {
		p.errorf("%v", err)
		return
	}
	gp, err := goobj.Parse(f, pkgpath)
	f.Close()
	if err != nil {
		p.errorf("reading %s: %v", file, err)
		return
	}

	// TODO(rsc): Change debug/goobj to record package name as gp.Name.
	// TODO(rsc): If pkgpath == "main", check that gp.Name == "main".

	pkg.Package = gp

	for _, gs := range gp.Syms {
		// TODO(rsc): Fix file format instead of this workaround.
		if gs.Data.Size > 0 {
			switch gs.Kind {
			case goobj.SBSS:
				gs.Kind = goobj.SDATA
			case goobj.SNOPTRBSS:
				gs.Kind = goobj.SNOPTRDATA
			}
		}

		if gs.Version != 0 {
			gs.Version += p.MaxVersion
		}
		for i := range gs.Reloc {
			r := &gs.Reloc[i]
			if r.Sym.Version != 0 {
				r.Sym.Version += p.MaxVersion
			}
			if p.Syms[r.Sym] != nil {
				p.Missing[r.Sym] = true
			}
		}
		if old := p.Syms[gs.SymID]; old != nil {
			p.errorf("symbol %s defined in both %s and %s", old.Package.File, file)
			continue
		}
		s := &Sym{
			Sym:     gs,
			Package: pkg,
		}
		pkg.Syms = append(pkg.Syms, s)
		p.Syms[gs.SymID] = s
		delete(p.Missing, gs.SymID)
	}
	p.MaxVersion += pkg.MaxVersion

	for i, pkgpath := range pkg.Imports {
		// TODO(rsc): Fix file format to drop .a from recorded import path.
		pkgpath = strings.TrimSuffix(pkgpath, ".a")
		pkg.Imports[i] = pkgpath

		p.scanImport(pkgpath)
	}
}

// scanImport finds the object file for the given import path and then scans it.
func (p *Prog) scanImport(pkgpath string) {
	if p.Packages[pkgpath] != nil {
		return // already loaded
	}

	// TODO(rsc): Implement correct search to find file.
	p.scanFile(pkgpath, "/Users/rsc/rscgo/pkg/darwin_amd64/"+pkgpath+".a")
}
