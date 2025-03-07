// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"go/build"
	"go/build/constraint"
	"go/token"
	"internal/godebug"
	"internal/goroot"
	"path"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"
	"unsafe"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/imports"
	"cmd/go/internal/str"
	"cmd/internal/par"
)

// enabled is used to flag off the behavior of the module index on tip, for debugging.
var enabled = godebug.New("#goindex").Value() != "0"

// Module represents and encoded module index file. It is used to
// do the equivalent of build.Import of packages in the module and answer other
// questions based on the index file's data.
type Module struct {
	modroot string
	d       *decoder
	n       int // number of packages
}

// moduleHash returns an ActionID corresponding to the state of the module
// located at filesystem path modroot.
func moduleHash(modroot string, ismodcache bool) (cache.ActionID, error) {
	// We expect modules stored within the module cache to be checksummed and
	// immutable, and we expect released modules within GOROOT to change only
	// infrequently (when the Go version changes).
	if !ismodcache {
		// The contents of this module may change over time. We don't want to pay
		// the cost to detect changes and re-index whenever they occur, so just
		// don't index it at all.
		//
		// Note that this is true even for modules in GOROOT/src: non-release builds
		// of the Go toolchain may have arbitrary development changes on top of the
		// commit reported by runtime.Version, or could be completely artificial due
		// to lacking a `git` binary (like "devel gomote.XXXXX", as synthesized by
		// "gomote push" as of 2022-06-15). (Release builds shouldn't have
		// modifications, but we don't want to use a behavior for releases that we
		// haven't tested during development.)
		return cache.ActionID{}, ErrNotIndexed
	}

	h := cache.NewHash("moduleIndex")
	// TODO(bcmills): Since modules in the index are checksummed, we could
	// probably improve the cache hit rate by keying off of the module
	// path@version (perhaps including the checksum?) instead of the module root
	// directory.
	fmt.Fprintf(h, "module index %s %s %v\n", runtime.Version(), indexVersion, modroot)
	return h.Sum(), nil
}

const modTimeCutoff = 2 * time.Second

// dirHash returns an ActionID corresponding to the state of the package
// located at filesystem path pkgdir.
func dirHash(modroot, pkgdir string) (cache.ActionID, error) {
	h := cache.NewHash("moduleIndex")
	fmt.Fprintf(h, "modroot %s\n", modroot)
	fmt.Fprintf(h, "package %s %s %v\n", runtime.Version(), indexVersion, pkgdir)
	dirs, err := fsys.ReadDir(pkgdir)
	if err != nil {
		// pkgdir might not be a directory. give up on hashing.
		return cache.ActionID{}, ErrNotIndexed
	}
	cutoff := time.Now().Add(-modTimeCutoff)
	for _, d := range dirs {
		if d.IsDir() {
			continue
		}

		if !d.Type().IsRegular() {
			return cache.ActionID{}, ErrNotIndexed
		}
		// To avoid problems for very recent files where a new
		// write might not change the mtime due to file system
		// mtime precision, reject caching if a file was read that
		// is less than modTimeCutoff old.
		//
		// This is the same strategy used for hashing test inputs.
		// See hashOpen in cmd/go/internal/test/test.go for the
		// corresponding code.
		info, err := d.Info()
		if err != nil {
			return cache.ActionID{}, ErrNotIndexed
		}
		if info.ModTime().After(cutoff) {
			return cache.ActionID{}, ErrNotIndexed
		}

		fmt.Fprintf(h, "file %v %v %v\n", info.Name(), info.ModTime(), info.Size())
	}
	return h.Sum(), nil
}

var ErrNotIndexed = errors.New("not in module index")

var (
	errDisabled           = fmt.Errorf("%w: module indexing disabled", ErrNotIndexed)
	errNotFromModuleCache = fmt.Errorf("%w: not from module cache", ErrNotIndexed)
	errFIPS140            = fmt.Errorf("%w: fips140 snapshots not indexed", ErrNotIndexed)
)

// GetPackage returns the IndexPackage for the directory at the given path.
// It will return ErrNotIndexed if the directory should be read without
// using the index, for instance because the index is disabled, or the package
// is not in a module.
func GetPackage(modroot, pkgdir string) (*IndexPackage, error) {
	mi, err := GetModule(modroot)
	if err == nil {
		return mi.Package(relPath(pkgdir, modroot)), nil
	}
	if !errors.Is(err, errNotFromModuleCache) {
		return nil, err
	}
	if cfg.BuildContext.Compiler == "gccgo" && str.HasPathPrefix(modroot, cfg.GOROOTsrc) {
		return nil, err // gccgo has no sources for GOROOT packages.
	}
	// The pkgdir for fips140 has been replaced in the fsys overlay,
	// but the module index does not see that. Do not try to use the module index.
	if strings.Contains(filepath.ToSlash(pkgdir), "internal/fips140/v") {
		return nil, errFIPS140
	}
	modroot = filepath.Clean(modroot)
	pkgdir = filepath.Clean(pkgdir)
	return openIndexPackage(modroot, pkgdir)
}

// GetModule returns the Module for the given modroot.
// It will return ErrNotIndexed if the directory should be read without
// using the index, for instance because the index is disabled, or the package
// is not in a module.
func GetModule(modroot string) (*Module, error) {
	dir, _, _ := cache.DefaultDir()
	if !enabled || dir == "off" {
		return nil, errDisabled
	}
	if modroot == "" {
		panic("modindex.GetPackage called with empty modroot")
	}
	if cfg.BuildMod == "vendor" {
		// Even if the main module is in the module cache,
		// its vendored dependencies are not loaded from their
		// usual cached locations.
		return nil, errNotFromModuleCache
	}
	modroot = filepath.Clean(modroot)
	if str.HasFilePathPrefix(modroot, cfg.GOROOTsrc) || !str.HasFilePathPrefix(modroot, cfg.GOMODCACHE) {
		return nil, errNotFromModuleCache
	}
	return openIndexModule(modroot, true)
}

var mcache par.ErrCache[string, *Module]

// openIndexModule returns the module index for modPath.
// It will return ErrNotIndexed if the module can not be read
// using the index because it contains symlinks.
func openIndexModule(modroot string, ismodcache bool) (*Module, error) {
	return mcache.Do(modroot, func() (*Module, error) {
		fsys.Trace("openIndexModule", modroot)
		id, err := moduleHash(modroot, ismodcache)
		if err != nil {
			return nil, err
		}
		data, _, opened, err := cache.GetMmap(cache.Default(), id)
		if err != nil {
			// Couldn't read from modindex. Assume we couldn't read from
			// the index because the module hasn't been indexed yet.
			// But double check on Windows that we haven't opened the file yet,
			// because once mmap opens the file, we can't close it, and
			// Windows won't let us open an already opened file.
			data, err = indexModule(modroot)
			if err != nil {
				return nil, err
			}
			if runtime.GOOS != "windows" || !opened {
				if err = cache.PutBytes(cache.Default(), id, data); err != nil {
					return nil, err
				}
			}
		}
		mi, err := fromBytes(modroot, data)
		if err != nil {
			return nil, err
		}
		return mi, nil
	})
}

var pcache par.ErrCache[[2]string, *IndexPackage]

func openIndexPackage(modroot, pkgdir string) (*IndexPackage, error) {
	return pcache.Do([2]string{modroot, pkgdir}, func() (*IndexPackage, error) {
		fsys.Trace("openIndexPackage", pkgdir)
		id, err := dirHash(modroot, pkgdir)
		if err != nil {
			return nil, err
		}
		data, _, opened, err := cache.GetMmap(cache.Default(), id)
		if err != nil {
			// Couldn't read from index. Assume we couldn't read from
			// the index because the package hasn't been indexed yet.
			// But double check on Windows that we haven't opened the file yet,
			// because once mmap opens the file, we can't close it, and
			// Windows won't let us open an already opened file.
			data = indexPackage(modroot, pkgdir)
			if runtime.GOOS != "windows" || !opened {
				if err = cache.PutBytes(cache.Default(), id, data); err != nil {
					return nil, err
				}
			}
		}
		pkg, err := packageFromBytes(modroot, data)
		if err != nil {
			return nil, err
		}
		return pkg, nil
	})
}

var errCorrupt = errors.New("corrupt index")

// protect marks the start of a large section of code that accesses the index.
// It should be used as:
//
//	defer unprotect(protect, &err)
//
// It should not be used for trivial accesses which would be
// dwarfed by the overhead of the defer.
func protect() bool {
	return debug.SetPanicOnFault(true)
}

var isTest = false

// unprotect marks the end of a large section of code that accesses the index.
// It should be used as:
//
//	defer unprotect(protect, &err)
//
// end looks for panics due to errCorrupt or bad mmap accesses.
// When it finds them, it adds explanatory text, consumes the panic, and sets *errp instead.
// If errp is nil, end adds the explanatory text but then calls base.Fatalf.
func unprotect(old bool, errp *error) {
	// SetPanicOnFault's errors _may_ satisfy this interface. Even though it's not guaranteed
	// that all its errors satisfy this interface, we'll only check for these errors so that
	// we don't suppress panics that could have been produced from other sources.
	type addrer interface {
		Addr() uintptr
	}

	debug.SetPanicOnFault(old)

	if e := recover(); e != nil {
		if _, ok := e.(addrer); ok || e == errCorrupt {
			// This panic was almost certainly caused by SetPanicOnFault or our panic(errCorrupt).
			err := fmt.Errorf("error reading module index: %v", e)
			if errp != nil {
				*errp = err
				return
			}
			if isTest {
				panic(err)
			}
			base.Fatalf("%v", err)
		}
		// The panic was likely not caused by SetPanicOnFault.
		panic(e)
	}
}

// fromBytes returns a *Module given the encoded representation.
func fromBytes(moddir string, data []byte) (m *Module, err error) {
	if !enabled {
		panic("use of index")
	}

	defer unprotect(protect(), &err)

	if !bytes.HasPrefix(data, []byte(indexVersion+"\n")) {
		return nil, errCorrupt
	}

	const hdr = len(indexVersion + "\n")
	d := &decoder{data: data}
	str := d.intAt(hdr)
	if str < hdr+8 || len(d.data) < str {
		return nil, errCorrupt
	}
	d.data, d.str = data[:str], d.data[str:]
	// Check that string table looks valid.
	// First string is empty string (length 0),
	// and we leave a marker byte 0xFF at the end
	// just to make sure that the file is not truncated.
	if len(d.str) == 0 || d.str[0] != 0 || d.str[len(d.str)-1] != 0xFF {
		return nil, errCorrupt
	}

	n := d.intAt(hdr + 4)
	if n < 0 || n > (len(d.data)-8)/8 {
		return nil, errCorrupt
	}

	m = &Module{
		moddir,
		d,
		n,
	}
	return m, nil
}

// packageFromBytes returns a *IndexPackage given the encoded representation.
func packageFromBytes(modroot string, data []byte) (p *IndexPackage, err error) {
	m, err := fromBytes(modroot, data)
	if err != nil {
		return nil, err
	}
	if m.n != 1 {
		return nil, fmt.Errorf("corrupt single-package index")
	}
	return m.pkg(0), nil
}

// pkgDir returns the dir string of the i'th package in the index.
func (m *Module) pkgDir(i int) string {
	if i < 0 || i >= m.n {
		panic(errCorrupt)
	}
	return m.d.stringAt(12 + 8 + 8*i)
}

// pkgOff returns the offset of the data for the i'th package in the index.
func (m *Module) pkgOff(i int) int {
	if i < 0 || i >= m.n {
		panic(errCorrupt)
	}
	return m.d.intAt(12 + 8 + 8*i + 4)
}

// Walk calls f for each package in the index, passing the path to that package relative to the module root.
func (m *Module) Walk(f func(path string)) {
	defer unprotect(protect(), nil)
	for i := 0; i < m.n; i++ {
		f(m.pkgDir(i))
	}
}

// relPath returns the path relative to the module's root.
func relPath(path, modroot string) string {
	return str.TrimFilePathPrefix(filepath.Clean(path), filepath.Clean(modroot))
}

var installgorootAll = godebug.New("installgoroot").Value() == "all"

// Import is the equivalent of build.Import given the information in Module.
func (rp *IndexPackage) Import(bctxt build.Context, mode build.ImportMode) (p *build.Package, err error) {
	defer unprotect(protect(), &err)

	ctxt := (*Context)(&bctxt)

	p = &build.Package{}

	p.ImportPath = "."
	p.Dir = filepath.Join(rp.modroot, rp.dir)

	var pkgerr error
	switch ctxt.Compiler {
	case "gccgo", "gc":
	default:
		// Save error for end of function.
		pkgerr = fmt.Errorf("import %q: unknown compiler %q", p.Dir, ctxt.Compiler)
	}

	if p.Dir == "" {
		return p, fmt.Errorf("import %q: import of unknown directory", p.Dir)
	}

	// goroot and gopath
	inTestdata := func(sub string) bool {
		return strings.Contains(sub, "/testdata/") || strings.HasSuffix(sub, "/testdata") || str.HasPathPrefix(sub, "testdata")
	}
	var pkga string
	if !inTestdata(rp.dir) {
		// In build.go, p.Root should only be set in the non-local-import case, or in
		// GOROOT or GOPATH. Since module mode only calls Import with path set to "."
		// and the module index doesn't apply outside modules, the GOROOT case is
		// the only case where p.Root needs to be set.
		if ctxt.GOROOT != "" && str.HasFilePathPrefix(p.Dir, cfg.GOROOTsrc) && p.Dir != cfg.GOROOTsrc {
			p.Root = ctxt.GOROOT
			p.Goroot = true
			modprefix := str.TrimFilePathPrefix(rp.modroot, cfg.GOROOTsrc)
			p.ImportPath = rp.dir
			if modprefix != "" {
				p.ImportPath = filepath.Join(modprefix, p.ImportPath)
			}

			// Set GOROOT-specific fields (sometimes for modules in a GOPATH directory).
			// The fields set below (SrcRoot, PkgRoot, BinDir, PkgTargetRoot, and PkgObj)
			// are only set in build.Import if p.Root != "".
			var pkgtargetroot string
			suffix := ""
			if ctxt.InstallSuffix != "" {
				suffix = "_" + ctxt.InstallSuffix
			}
			switch ctxt.Compiler {
			case "gccgo":
				pkgtargetroot = "pkg/gccgo_" + ctxt.GOOS + "_" + ctxt.GOARCH + suffix
				dir, elem := path.Split(p.ImportPath)
				pkga = pkgtargetroot + "/" + dir + "lib" + elem + ".a"
			case "gc":
				pkgtargetroot = "pkg/" + ctxt.GOOS + "_" + ctxt.GOARCH + suffix
				pkga = pkgtargetroot + "/" + p.ImportPath + ".a"
			}
			p.SrcRoot = ctxt.joinPath(p.Root, "src")
			p.PkgRoot = ctxt.joinPath(p.Root, "pkg")
			p.BinDir = ctxt.joinPath(p.Root, "bin")
			if pkga != "" {
				// Always set PkgTargetRoot. It might be used when building in shared
				// mode.
				p.PkgTargetRoot = ctxt.joinPath(p.Root, pkgtargetroot)

				// Set the install target if applicable.
				if !p.Goroot || (installgorootAll && p.ImportPath != "unsafe" && p.ImportPath != "builtin") {
					p.PkgObj = ctxt.joinPath(p.Root, pkga)
				}
			}
		}
	}

	if rp.error != nil {
		if errors.Is(rp.error, errCannotFindPackage) && ctxt.Compiler == "gccgo" && p.Goroot {
			return p, nil
		}
		return p, rp.error
	}

	if mode&build.FindOnly != 0 {
		return p, pkgerr
	}

	// We need to do a second round of bad file processing.
	var badGoError error
	badGoFiles := make(map[string]bool)
	badGoFile := func(name string, err error) {
		if badGoError == nil {
			badGoError = err
		}
		if !badGoFiles[name] {
			p.InvalidGoFiles = append(p.InvalidGoFiles, name)
			badGoFiles[name] = true
		}
	}

	var Sfiles []string // files with ".S"(capital S)/.sx(capital s equivalent for case insensitive filesystems)
	var firstFile string
	embedPos := make(map[string][]token.Position)
	testEmbedPos := make(map[string][]token.Position)
	xTestEmbedPos := make(map[string][]token.Position)
	importPos := make(map[string][]token.Position)
	testImportPos := make(map[string][]token.Position)
	xTestImportPos := make(map[string][]token.Position)
	allTags := make(map[string]bool)
	for _, tf := range rp.sourceFiles {
		name := tf.name()
		// Check errors for go files and call badGoFiles to put them in
		// InvalidGoFiles if they do have an error.
		if strings.HasSuffix(name, ".go") {
			if error := tf.error(); error != "" {
				badGoFile(name, errors.New(tf.error()))
				continue
			} else if parseError := tf.parseError(); parseError != "" {
				badGoFile(name, parseErrorFromString(tf.parseError()))
				// Fall through: we still want to list files with parse errors.
			}
		}

		var shouldBuild = true
		if !ctxt.goodOSArchFile(name, allTags) && !ctxt.UseAllFiles {
			shouldBuild = false
		} else if goBuildConstraint := tf.goBuildConstraint(); goBuildConstraint != "" {
			x, err := constraint.Parse(goBuildConstraint)
			if err != nil {
				return p, fmt.Errorf("%s: parsing //go:build line: %v", name, err)
			}
			shouldBuild = ctxt.eval(x, allTags)
		} else if plusBuildConstraints := tf.plusBuildConstraints(); len(plusBuildConstraints) > 0 {
			for _, text := range plusBuildConstraints {
				if x, err := constraint.Parse(text); err == nil {
					if !ctxt.eval(x, allTags) {
						shouldBuild = false
					}
				}
			}
		}

		ext := nameExt(name)
		if !shouldBuild || tf.ignoreFile() {
			if ext == ".go" {
				p.IgnoredGoFiles = append(p.IgnoredGoFiles, name)
			} else if fileListForExt(p, ext) != nil {
				p.IgnoredOtherFiles = append(p.IgnoredOtherFiles, name)
			}
			continue
		}

		// Going to save the file. For non-Go files, can stop here.
		switch ext {
		case ".go":
			// keep going
		case ".S", ".sx":
			// special case for cgo, handled at end
			Sfiles = append(Sfiles, name)
			continue
		default:
			if list := fileListForExt(p, ext); list != nil {
				*list = append(*list, name)
			}
			continue
		}

		pkg := tf.pkgName()
		if pkg == "documentation" {
			p.IgnoredGoFiles = append(p.IgnoredGoFiles, name)
			continue
		}
		isTest := strings.HasSuffix(name, "_test.go")
		isXTest := false
		if isTest && strings.HasSuffix(tf.pkgName(), "_test") && p.Name != tf.pkgName() {
			isXTest = true
			pkg = pkg[:len(pkg)-len("_test")]
		}

		if !isTest && tf.binaryOnly() {
			p.BinaryOnly = true
		}

		if p.Name == "" {
			p.Name = pkg
			firstFile = name
		} else if pkg != p.Name {
			// TODO(#45999): The choice of p.Name is arbitrary based on file iteration
			// order. Instead of resolving p.Name arbitrarily, we should clear out the
			// existing Name and mark the existing files as also invalid.
			badGoFile(name, &MultiplePackageError{
				Dir:      p.Dir,
				Packages: []string{p.Name, pkg},
				Files:    []string{firstFile, name},
			})
		}
		// Grab the first package comment as docs, provided it is not from a test file.
		if p.Doc == "" && !isTest && !isXTest {
			if synopsis := tf.synopsis(); synopsis != "" {
				p.Doc = synopsis
			}
		}

		// Record Imports and information about cgo.
		isCgo := false
		imports := tf.imports()
		for _, imp := range imports {
			if imp.path == "C" {
				if isTest {
					badGoFile(name, fmt.Errorf("use of cgo in test %s not supported", name))
					continue
				}
				isCgo = true
			}
		}
		if directives := tf.cgoDirectives(); directives != "" {
			if err := ctxt.saveCgo(name, p, directives); err != nil {
				badGoFile(name, err)
			}
		}

		var fileList *[]string
		var importMap, embedMap map[string][]token.Position
		var directives *[]build.Directive
		switch {
		case isCgo:
			allTags["cgo"] = true
			if ctxt.CgoEnabled {
				fileList = &p.CgoFiles
				importMap = importPos
				embedMap = embedPos
				directives = &p.Directives
			} else {
				// Ignore Imports and Embeds from cgo files if cgo is disabled.
				fileList = &p.IgnoredGoFiles
			}
		case isXTest:
			fileList = &p.XTestGoFiles
			importMap = xTestImportPos
			embedMap = xTestEmbedPos
			directives = &p.XTestDirectives
		case isTest:
			fileList = &p.TestGoFiles
			importMap = testImportPos
			embedMap = testEmbedPos
			directives = &p.TestDirectives
		default:
			fileList = &p.GoFiles
			importMap = importPos
			embedMap = embedPos
			directives = &p.Directives
		}
		*fileList = append(*fileList, name)
		if importMap != nil {
			for _, imp := range imports {
				importMap[imp.path] = append(importMap[imp.path], imp.position)
			}
		}
		if embedMap != nil {
			for _, e := range tf.embeds() {
				embedMap[e.pattern] = append(embedMap[e.pattern], e.position)
			}
		}
		if directives != nil {
			*directives = append(*directives, tf.directives()...)
		}
	}

	p.EmbedPatterns, p.EmbedPatternPos = cleanDecls(embedPos)
	p.TestEmbedPatterns, p.TestEmbedPatternPos = cleanDecls(testEmbedPos)
	p.XTestEmbedPatterns, p.XTestEmbedPatternPos = cleanDecls(xTestEmbedPos)

	p.Imports, p.ImportPos = cleanDecls(importPos)
	p.TestImports, p.TestImportPos = cleanDecls(testImportPos)
	p.XTestImports, p.XTestImportPos = cleanDecls(xTestImportPos)

	for tag := range allTags {
		p.AllTags = append(p.AllTags, tag)
	}
	sort.Strings(p.AllTags)

	if len(p.CgoFiles) > 0 {
		p.SFiles = append(p.SFiles, Sfiles...)
		sort.Strings(p.SFiles)
	} else {
		p.IgnoredOtherFiles = append(p.IgnoredOtherFiles, Sfiles...)
		sort.Strings(p.IgnoredOtherFiles)
	}

	if badGoError != nil {
		return p, badGoError
	}
	if len(p.GoFiles)+len(p.CgoFiles)+len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 {
		return p, &build.NoGoError{Dir: p.Dir}
	}
	return p, pkgerr
}

// IsStandardPackage reports whether path is a standard package
// for the goroot and compiler using the module index if possible,
// and otherwise falling back to internal/goroot.IsStandardPackage
func IsStandardPackage(goroot_, compiler, path string) bool {
	if !enabled || compiler != "gc" {
		return goroot.IsStandardPackage(goroot_, compiler, path)
	}

	reldir := filepath.FromSlash(path) // relative dir path in module index for package
	modroot := filepath.Join(goroot_, "src")
	if str.HasFilePathPrefix(reldir, "cmd") {
		reldir = str.TrimFilePathPrefix(reldir, "cmd")
		modroot = filepath.Join(modroot, "cmd")
	}
	if pkg, err := GetPackage(modroot, filepath.Join(modroot, reldir)); err == nil {
		hasGo, err := pkg.IsGoDir()
		return err == nil && hasGo
	} else if errors.Is(err, ErrNotIndexed) {
		// Fall back because package isn't indexable. (Probably because
		// a file was modified recently)
		return goroot.IsStandardPackage(goroot_, compiler, path)
	}
	return false
}

// IsGoDir is the equivalent of fsys.IsGoDir using the information in the index.
func (rp *IndexPackage) IsGoDir() (_ bool, err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("error reading module index: %v", e)
		}
	}()
	for _, sf := range rp.sourceFiles {
		if strings.HasSuffix(sf.name(), ".go") {
			return true, nil
		}
	}
	return false, nil
}

// ScanDir implements imports.ScanDir using the information in the index.
func (rp *IndexPackage) ScanDir(tags map[string]bool) (sortedImports []string, sortedTestImports []string, err error) {
	// TODO(matloob) dir should eventually be relative to indexed directory
	// TODO(matloob): skip reading raw package and jump straight to data we need?

	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("error reading module index: %v", e)
		}
	}()

	imports_ := make(map[string]bool)
	testImports := make(map[string]bool)
	numFiles := 0

Files:
	for _, sf := range rp.sourceFiles {
		name := sf.name()
		if strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".go") || !imports.MatchFile(name, tags) {
			continue
		}

		// The following section exists for backwards compatibility reasons:
		// scanDir ignores files with import "C" when collecting the list
		// of imports unless the "cgo" tag is provided. The following comment
		// is copied from the original.
		//
		// import "C" is implicit requirement of cgo tag.
		// When listing files on the command line (explicitFiles=true)
		// we do not apply build tag filtering but we still do apply
		// cgo filtering, so no explicitFiles check here.
		// Why? Because we always have, and it's not worth breaking
		// that behavior now.
		imps := sf.imports() // TODO(matloob): directly read import paths to avoid the extra strings?
		for _, imp := range imps {
			if imp.path == "C" && !tags["cgo"] && !tags["*"] {
				continue Files
			}
		}

		if !shouldBuild(sf, tags) {
			continue
		}
		numFiles++
		m := imports_
		if strings.HasSuffix(name, "_test.go") {
			m = testImports
		}
		for _, p := range imps {
			m[p.path] = true
		}
	}
	if numFiles == 0 {
		return nil, nil, imports.ErrNoGo
	}
	return keys(imports_), keys(testImports), nil
}

func keys(m map[string]bool) []string {
	list := make([]string, 0, len(m))
	for k := range m {
		list = append(list, k)
	}
	sort.Strings(list)
	return list
}

// implements imports.ShouldBuild in terms of an index sourcefile.
func shouldBuild(sf *sourceFile, tags map[string]bool) bool {
	if goBuildConstraint := sf.goBuildConstraint(); goBuildConstraint != "" {
		x, err := constraint.Parse(goBuildConstraint)
		if err != nil {
			return false
		}
		return imports.Eval(x, tags, true)
	}

	plusBuildConstraints := sf.plusBuildConstraints()
	for _, text := range plusBuildConstraints {
		if x, err := constraint.Parse(text); err == nil {
			if !imports.Eval(x, tags, true) {
				return false
			}
		}
	}

	return true
}

// IndexPackage holds the information in the index
// needed to load a package in a specific directory.
type IndexPackage struct {
	error error
	dir   string // directory of the package relative to the modroot

	modroot string

	// Source files
	sourceFiles []*sourceFile
}

var errCannotFindPackage = errors.New("cannot find package")

// Package and returns finds the package with the given path (relative to the module root).
// If the package does not exist, Package returns an IndexPackage that will return an
// appropriate error from its methods.
func (m *Module) Package(path string) *IndexPackage {
	defer unprotect(protect(), nil)

	i, ok := sort.Find(m.n, func(i int) int {
		return strings.Compare(path, m.pkgDir(i))
	})
	if !ok {
		return &IndexPackage{error: fmt.Errorf("%w %q in:\n\t%s", errCannotFindPackage, path, filepath.Join(m.modroot, path))}
	}
	return m.pkg(i)
}

// pkg returns the i'th IndexPackage in m.
func (m *Module) pkg(i int) *IndexPackage {
	r := m.d.readAt(m.pkgOff(i))
	p := new(IndexPackage)
	if errstr := r.string(); errstr != "" {
		p.error = errors.New(errstr)
	}
	p.dir = r.string()
	p.sourceFiles = make([]*sourceFile, r.int())
	for i := range p.sourceFiles {
		p.sourceFiles[i] = &sourceFile{
			d:   m.d,
			pos: r.int(),
		}
	}
	p.modroot = m.modroot
	return p
}

// sourceFile represents the information of a given source file in the module index.
type sourceFile struct {
	d               *decoder // encoding of this source file
	pos             int      // start of sourceFile encoding in d
	onceReadImports sync.Once
	savedImports    []rawImport // saved imports so that they're only read once
}

// Offsets for fields in the sourceFile.
const (
	sourceFileError = 4 * iota
	sourceFileParseError
	sourceFileSynopsis
	sourceFileName
	sourceFilePkgName
	sourceFileIgnoreFile
	sourceFileBinaryOnly
	sourceFileCgoDirectives
	sourceFileGoBuildConstraint
	sourceFileNumPlusBuildConstraints
)

func (sf *sourceFile) error() string {
	return sf.d.stringAt(sf.pos + sourceFileError)
}
func (sf *sourceFile) parseError() string {
	return sf.d.stringAt(sf.pos + sourceFileParseError)
}
func (sf *sourceFile) synopsis() string {
	return sf.d.stringAt(sf.pos + sourceFileSynopsis)
}
func (sf *sourceFile) name() string {
	return sf.d.stringAt(sf.pos + sourceFileName)
}
func (sf *sourceFile) pkgName() string {
	return sf.d.stringAt(sf.pos + sourceFilePkgName)
}
func (sf *sourceFile) ignoreFile() bool {
	return sf.d.boolAt(sf.pos + sourceFileIgnoreFile)
}
func (sf *sourceFile) binaryOnly() bool {
	return sf.d.boolAt(sf.pos + sourceFileBinaryOnly)
}
func (sf *sourceFile) cgoDirectives() string {
	return sf.d.stringAt(sf.pos + sourceFileCgoDirectives)
}
func (sf *sourceFile) goBuildConstraint() string {
	return sf.d.stringAt(sf.pos + sourceFileGoBuildConstraint)
}

func (sf *sourceFile) plusBuildConstraints() []string {
	pos := sf.pos + sourceFileNumPlusBuildConstraints
	n := sf.d.intAt(pos)
	pos += 4
	ret := make([]string, n)
	for i := 0; i < n; i++ {
		ret[i] = sf.d.stringAt(pos)
		pos += 4
	}
	return ret
}

func (sf *sourceFile) importsOffset() int {
	pos := sf.pos + sourceFileNumPlusBuildConstraints
	n := sf.d.intAt(pos)
	// each build constraint is 1 uint32
	return pos + 4 + n*4
}

func (sf *sourceFile) embedsOffset() int {
	pos := sf.importsOffset()
	n := sf.d.intAt(pos)
	// each import is 5 uint32s (string + tokpos)
	return pos + 4 + n*(4*5)
}

func (sf *sourceFile) directivesOffset() int {
	pos := sf.embedsOffset()
	n := sf.d.intAt(pos)
	// each embed is 5 uint32s (string + tokpos)
	return pos + 4 + n*(4*5)
}

func (sf *sourceFile) imports() []rawImport {
	sf.onceReadImports.Do(func() {
		importsOffset := sf.importsOffset()
		r := sf.d.readAt(importsOffset)
		numImports := r.int()
		ret := make([]rawImport, numImports)
		for i := 0; i < numImports; i++ {
			ret[i] = rawImport{r.string(), r.tokpos()}
		}
		sf.savedImports = ret
	})
	return sf.savedImports
}

func (sf *sourceFile) embeds() []embed {
	embedsOffset := sf.embedsOffset()
	r := sf.d.readAt(embedsOffset)
	numEmbeds := r.int()
	ret := make([]embed, numEmbeds)
	for i := range ret {
		ret[i] = embed{r.string(), r.tokpos()}
	}
	return ret
}

func (sf *sourceFile) directives() []build.Directive {
	directivesOffset := sf.directivesOffset()
	r := sf.d.readAt(directivesOffset)
	numDirectives := r.int()
	ret := make([]build.Directive, numDirectives)
	for i := range ret {
		ret[i] = build.Directive{Text: r.string(), Pos: r.tokpos()}
	}
	return ret
}

func asString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

// A decoder helps decode the index format.
type decoder struct {
	data []byte // data after header
	str  []byte // string table
}

// intAt returns the int at the given offset in d.data.
func (d *decoder) intAt(off int) int {
	if off < 0 || len(d.data)-off < 4 {
		panic(errCorrupt)
	}
	i := binary.LittleEndian.Uint32(d.data[off : off+4])
	if int32(i)>>31 != 0 {
		panic(errCorrupt)
	}
	return int(i)
}

// boolAt returns the bool at the given offset in d.data.
func (d *decoder) boolAt(off int) bool {
	return d.intAt(off) != 0
}

// stringAt returns the string pointed at by the int at the given offset in d.data.
func (d *decoder) stringAt(off int) string {
	return d.stringTableAt(d.intAt(off))
}

// stringTableAt returns the string at the given offset in the string table d.str.
func (d *decoder) stringTableAt(off int) string {
	if off < 0 || off >= len(d.str) {
		panic(errCorrupt)
	}
	s := d.str[off:]
	v, n := binary.Uvarint(s)
	if n <= 0 || v > uint64(len(s[n:])) {
		panic(errCorrupt)
	}
	return asString(s[n : n+int(v)])
}

// A reader reads sequential fields from a section of the index format.
type reader struct {
	d   *decoder
	pos int
}

// readAt returns a reader starting at the given position in d.
func (d *decoder) readAt(pos int) *reader {
	return &reader{d, pos}
}

// int reads the next int.
func (r *reader) int() int {
	i := r.d.intAt(r.pos)
	r.pos += 4
	return i
}

// string reads the next string.
func (r *reader) string() string {
	return r.d.stringTableAt(r.int())
}

// bool reads the next bool.
func (r *reader) bool() bool {
	return r.int() != 0
}

// tokpos reads the next token.Position.
func (r *reader) tokpos() token.Position {
	return token.Position{
		Filename: r.string(),
		Offset:   r.int(),
		Line:     r.int(),
		Column:   r.int(),
	}
}
