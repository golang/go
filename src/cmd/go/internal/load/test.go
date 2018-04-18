// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/str"
	"go/build"
	"go/token"
)

// TestPackagesFor returns package structs ptest, the package p plus
// its test files, and pxtest, the external tests of package p.
// pxtest may be nil. If there are no test files, forceTest decides
// whether this returns a new package struct or just returns p.
func TestPackagesFor(p *Package, forceTest bool) (ptest, pxtest *Package, err error) {
	var imports, ximports []*Package
	var stk ImportStack
	stk.Push(p.ImportPath + " (test)")
	rawTestImports := str.StringList(p.TestImports)
	for i, path := range p.TestImports {
		p1 := LoadImport(path, p.Dir, p, &stk, p.Internal.Build.TestImportPos[path], UseVendor)
		if p1.Error != nil {
			return nil, nil, p1.Error
		}
		if len(p1.DepsErrors) > 0 {
			err := p1.DepsErrors[0]
			err.Pos = "" // show full import stack
			return nil, nil, err
		}
		if str.Contains(p1.Deps, p.ImportPath) || p1.ImportPath == p.ImportPath {
			// Same error that loadPackage returns (via reusePackage) in pkg.go.
			// Can't change that code, because that code is only for loading the
			// non-test copy of a package.
			err := &PackageError{
				ImportStack:   testImportStack(stk[0], p1, p.ImportPath),
				Err:           "import cycle not allowed in test",
				IsImportCycle: true,
			}
			return nil, nil, err
		}
		p.TestImports[i] = p1.ImportPath
		imports = append(imports, p1)
	}
	stk.Pop()
	stk.Push(p.ImportPath + "_test")
	pxtestNeedsPtest := false
	rawXTestImports := str.StringList(p.XTestImports)
	for i, path := range p.XTestImports {
		p1 := LoadImport(path, p.Dir, p, &stk, p.Internal.Build.XTestImportPos[path], UseVendor)
		if p1.Error != nil {
			return nil, nil, p1.Error
		}
		if len(p1.DepsErrors) > 0 {
			err := p1.DepsErrors[0]
			err.Pos = "" // show full import stack
			return nil, nil, err
		}
		if p1.ImportPath == p.ImportPath {
			pxtestNeedsPtest = true
		} else {
			ximports = append(ximports, p1)
		}
		p.XTestImports[i] = p1.ImportPath
	}
	stk.Pop()

	// Test package.
	if len(p.TestGoFiles) > 0 || forceTest {
		ptest = new(Package)
		*ptest = *p
		ptest.GoFiles = nil
		ptest.GoFiles = append(ptest.GoFiles, p.GoFiles...)
		ptest.GoFiles = append(ptest.GoFiles, p.TestGoFiles...)
		ptest.Target = ""
		// Note: The preparation of the vet config requires that common
		// indexes in ptest.Imports, ptest.Internal.Imports, and ptest.Internal.RawImports
		// all line up (but RawImports can be shorter than the others).
		// That is, for 0 ≤ i < len(RawImports),
		// RawImports[i] is the import string in the program text,
		// Imports[i] is the expanded import string (vendoring applied or relative path expanded away),
		// and Internal.Imports[i] is the corresponding *Package.
		// Any implicitly added imports appear in Imports and Internal.Imports
		// but not RawImports (because they were not in the source code).
		// We insert TestImports, imports, and rawTestImports at the start of
		// these lists to preserve the alignment.
		ptest.Imports = str.StringList(p.TestImports, p.Imports)
		ptest.Internal.Imports = append(imports, p.Internal.Imports...)
		ptest.Internal.RawImports = str.StringList(rawTestImports, p.Internal.RawImports)
		ptest.Internal.ForceLibrary = true
		ptest.Internal.Build = new(build.Package)
		*ptest.Internal.Build = *p.Internal.Build
		m := map[string][]token.Position{}
		for k, v := range p.Internal.Build.ImportPos {
			m[k] = append(m[k], v...)
		}
		for k, v := range p.Internal.Build.TestImportPos {
			m[k] = append(m[k], v...)
		}
		ptest.Internal.Build.ImportPos = m
	} else {
		ptest = p
	}

	// External test package.
	if len(p.XTestGoFiles) > 0 {
		pxtest = &Package{
			PackagePublic: PackagePublic{
				Name:       p.Name + "_test",
				ImportPath: p.ImportPath + "_test",
				Root:       p.Root,
				Dir:        p.Dir,
				GoFiles:    p.XTestGoFiles,
				Imports:    p.XTestImports,
			},
			Internal: PackageInternal{
				LocalPrefix: p.Internal.LocalPrefix,
				Build: &build.Package{
					ImportPos: p.Internal.Build.XTestImportPos,
				},
				Imports:    ximports,
				RawImports: rawXTestImports,

				Asmflags:   p.Internal.Asmflags,
				Gcflags:    p.Internal.Gcflags,
				Ldflags:    p.Internal.Ldflags,
				Gccgoflags: p.Internal.Gccgoflags,
			},
		}
		if pxtestNeedsPtest {
			pxtest.Internal.Imports = append(pxtest.Internal.Imports, ptest)
		}
	}

	if p != ptest && pxtest != nil {
		// We have made modifications to the package p being tested
		// and are rebuilding p (as ptest).
		// Arrange to rebuild all packages q such that
		// pxtest depends on q and q depends on p.
		// This makes sure that q sees the modifications to p.
		// Strictly speaking, the rebuild is only necessary if the
		// modifications to p change its export metadata, but
		// determining that is a bit tricky, so we rebuild always.
		recompileForTest(p, ptest, pxtest)
	}

	return ptest, pxtest, nil
}

func testImportStack(top string, p *Package, target string) []string {
	stk := []string{top, p.ImportPath}
Search:
	for p.ImportPath != target {
		for _, p1 := range p.Internal.Imports {
			if p1.ImportPath == target || str.Contains(p1.Deps, target) {
				stk = append(stk, p1.ImportPath)
				p = p1
				continue Search
			}
		}
		// Can't happen, but in case it does...
		stk = append(stk, "<lost path to cycle>")
		break
	}
	return stk
}

func recompileForTest(preal, ptest, pxtest *Package) {
	// The "test copy" of preal is ptest.
	// For each package that depends on preal, make a "test copy"
	// that depends on ptest. And so on, up the dependency tree.
	testCopy := map[*Package]*Package{preal: ptest}
	// Only pxtest and its dependencies can legally depend on p.
	// If ptest or its dependencies depended on p, the dependency
	// would be circular.
	for _, p := range PackageList([]*Package{pxtest}) {
		if p == preal {
			continue
		}
		// Copy on write.
		didSplit := p == pxtest
		split := func() {
			if didSplit {
				return
			}
			didSplit = true
			if testCopy[p] != nil {
				panic("recompileForTest loop")
			}
			p1 := new(Package)
			testCopy[p] = p1
			*p1 = *p
			p1.Internal.Imports = make([]*Package, len(p.Internal.Imports))
			copy(p1.Internal.Imports, p.Internal.Imports)
			p = p1
			p.Target = ""
		}

		// Update p.Internal.Imports to use test copies.
		for i, imp := range p.Internal.Imports {
			if p1 := testCopy[imp]; p1 != nil && p1 != imp {
				split()
				p.Internal.Imports[i] = p1
			}
		}
	}
}
