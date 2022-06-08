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
	"internal/goroot"
	"internal/unsafeheader"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"unsafe"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/imports"
	"cmd/go/internal/par"
	"cmd/go/internal/str"
)

// enabled is used to flag off the behavior of the module index on tip.
// It will be removed before the release.
// TODO(matloob): Remove enabled once we have more confidence on the
// module index.
var enabled = func() bool {
	debug := strings.Split(os.Getenv("GODEBUG"), ",")
	for _, f := range debug {
		if f == "goindex=0" {
			return false
		}
	}
	return true
}()

// ModuleIndex represents and encoded module index file. It is used to
// do the equivalent of build.Import of packages in the module and answer other
// questions based on the index file's data.
type ModuleIndex struct {
	modroot      string
	od           offsetDecoder
	packages     map[string]int // offsets of each package
	packagePaths []string       // paths to package directories relative to modroot; these are the keys of packages
}

var fcache par.Cache

func moduleHash(modroot string, ismodcache bool) (cache.ActionID, error) {
	h := cache.NewHash("moduleIndex")
	fmt.Fprintf(h, "module index %s %s %v\n", runtime.Version(), indexVersion, modroot)
	if ismodcache || str.HasFilePathPrefix(modroot, cfg.GOROOT) {
		return h.Sum(), nil
	}
	// walkdir happens in deterministic order.
	err := fsys.Walk(modroot, func(path string, info fs.FileInfo, err error) error {
		if modroot == path {
			// Check for go.mod in root directory, and return ErrNotIndexed
			// if it doesn't exist. Outside the module cache, it's not a module
			// if it doesn't have a go.mod file.
		}
		if err := moduleWalkErr(modroot, path, info, err); err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}
		fmt.Fprintf(h, "file %v %v\n", info.Name(), info.ModTime())
		if info.Mode()&fs.ModeSymlink != 0 {
			targ, err := fsys.Stat(path)
			if err != nil {
				return err
			}
			fmt.Fprintf(h, "target %v %v\n", targ.Name(), targ.ModTime())
		}
		return nil
	})
	if err != nil {
		return cache.ActionID{}, err
	}
	return h.Sum(), nil
}

var modrootCache par.Cache

var ErrNotIndexed = errors.New("not in module index")

// Get returns the ModuleIndex for the module rooted at modroot.
// It will return ErrNotIndexed if the directory should be read without
// using the index, for instance because the index is disabled, or the packgae
// is not in a module.
func Get(modroot string) (*ModuleIndex, error) {
	if !enabled || cache.DefaultDir() == "off" || cfg.BuildMod == "vendor" {
		return nil, ErrNotIndexed
	}
	if modroot == "" {
		panic("modindex.Get called with empty modroot")
	}
	modroot = filepath.Clean(modroot)
	isModCache := str.HasFilePathPrefix(modroot, cfg.GOMODCACHE)
	return openIndex(modroot, isModCache)
}

// openIndex returns the module index for modPath.
// It will return ErrNotIndexed if the module can not be read
// using the index because it contains symlinks.
func openIndex(modroot string, ismodcache bool) (*ModuleIndex, error) {
	type result struct {
		mi  *ModuleIndex
		err error
	}
	r := fcache.Do(modroot, func() any {
		id, err := moduleHash(modroot, ismodcache)
		if err != nil {
			return result{nil, err}
		}
		data, _, err := cache.Default().GetMmap(id)
		if err != nil {
			// Couldn't read from modindex. Assume we couldn't read from
			// the index because the module hasn't been indexed yet.
			data, err = indexModule(modroot)
			if err != nil {
				return result{nil, err}
			}
			if err = cache.Default().PutBytes(id, data); err != nil {
				return result{nil, err}
			}
		}
		mi, err := fromBytes(modroot, data)
		if err != nil {
			return result{nil, err}
		}
		return result{mi, nil}
	}).(result)
	return r.mi, r.err
}

// fromBytes returns a *ModuleIndex given the encoded representation.
func fromBytes(moddir string, data []byte) (mi *ModuleIndex, err error) {
	if !enabled {
		panic("use of index")
	}

	// SetPanicOnFault's errors _may_ satisfy this interface. Even though it's not guaranteed
	// that all its errors satisfy this interface, we'll only check for these errors so that
	// we don't suppress panics that could have been produced from other sources.
	type addrer interface {
		Addr() uintptr
	}

	// set PanicOnFault to true so that we can catch errors on the initial reads of the slice,
	// in case it's mmapped (the common case).
	old := debug.SetPanicOnFault(true)
	defer func() {
		debug.SetPanicOnFault(old)
		if e := recover(); e != nil {
			if _, ok := e.(addrer); ok {
				// This panic was almost certainly caused by SetPanicOnFault.
				err = fmt.Errorf("error reading module index: %v", e)
				return
			}
			// The panic was likely not caused by SetPanicOnFault.
			panic(e)
		}
	}()

	gotVersion, unread, _ := bytes.Cut(data, []byte{'\n'})
	if string(gotVersion) != indexVersion {
		return nil, fmt.Errorf("bad index version string: %q", gotVersion)
	}
	stringTableOffset, unread := binary.LittleEndian.Uint32(unread[:4]), unread[4:]
	st := newStringTable(data[stringTableOffset:])
	d := decoder{unread, st}
	numPackages := d.int()

	packagePaths := make([]string, numPackages)
	for i := range packagePaths {
		packagePaths[i] = d.string()
	}
	packageOffsets := make([]int, numPackages)
	for i := range packageOffsets {
		packageOffsets[i] = d.int()
	}
	packages := make(map[string]int, numPackages)
	for i := range packagePaths {
		packages[packagePaths[i]] = packageOffsets[i]
	}

	return &ModuleIndex{
		moddir,
		offsetDecoder{data, st},
		packages,
		packagePaths,
	}, nil
}

// Returns a list of directory paths, relative to the modroot, for
// packages contained in the module index.
func (mi *ModuleIndex) Packages() []string {
	return mi.packagePaths
}

// RelPath returns the path relative to the module's root.
func (mi *ModuleIndex) RelPath(path string) string {
	return str.TrimFilePathPrefix(filepath.Clean(path), mi.modroot) // mi.modroot is already clean
}

// ImportPackage is the equivalent of build.Import given the information in ModuleIndex.
func (mi *ModuleIndex) Import(bctxt build.Context, relpath string, mode build.ImportMode) (p *build.Package, err error) {
	rp := mi.indexPackage(relpath)

	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("error reading module index: %v", e)
		}
	}()

	ctxt := (*Context)(&bctxt)

	p = &build.Package{}

	p.ImportPath = "."
	p.Dir = filepath.Join(mi.modroot, rp.dir)

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
	if !inTestdata(relpath) {
		// In build.go, p.Root should only be set in the non-local-import case, or in
		// GOROOT or GOPATH. Since module mode only calls Import with path set to "."
		// and the module index doesn't apply outside modules, the GOROOT case is
		// the only case where GOROOT needs to be set.
		// But: p.Root is actually set in the local-import case outside GOROOT, if
		// the directory is contained in GOPATH/src
		// TODO(#37015): fix that behavior in go/build and remove the gopath case
		// below.
		if ctxt.GOROOT != "" && str.HasFilePathPrefix(p.Dir, cfg.GOROOTsrc) && p.Dir != cfg.GOROOTsrc {
			p.Root = ctxt.GOROOT
			p.Goroot = true
			modprefix := str.TrimFilePathPrefix(mi.modroot, cfg.GOROOTsrc)
			p.ImportPath = relpath
			if modprefix != "" {
				p.ImportPath = filepath.Join(modprefix, p.ImportPath)
			}
		}
		for _, root := range ctxt.gopath() {
			// TODO(matloob): do we need to reimplement the conflictdir logic?

			// TODO(matloob): ctxt.hasSubdir evaluates symlinks, so it
			// can be slower than we'd like. Find out if we can drop this
			// logic before the release.
			if sub, ok := ctxt.hasSubdir(filepath.Join(root, "src"), p.Dir); ok {
				p.ImportPath = sub
				p.Root = root
			}
		}
	}
	if p.Root != "" {
		// Set GOROOT-specific fields (sometimes for modules in a GOPATH directory).
		// The fields set below (SrcRoot, PkgRoot, BinDir, PkgTargetRoot, and PkgObj)
		// are only set in build.Import if p.Root != "". As noted in the comment
		// on setting p.Root above, p.Root should only be set in the GOROOT case for the
		// set of packages we care about, but is also set for modules in a GOPATH src
		// directory.
		var pkgtargetroot string
		var pkga string
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
			p.PkgTargetRoot = ctxt.joinPath(p.Root, pkgtargetroot)
			p.PkgObj = ctxt.joinPath(p.Root, pkga)
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
	badFiles := make(map[string]bool)
	badFile := func(name string, err error) {
		if badGoError == nil {
			badGoError = err
		}
		if !badFiles[name] {
			p.InvalidGoFiles = append(p.InvalidGoFiles, name)
			badFiles[name] = true
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
		if error := tf.error(); error != "" {
			badFile(name, errors.New(tf.error()))
			continue
		} else if parseError := tf.parseError(); parseError != "" {
			badFile(name, parseErrorFromString(tf.parseError()))
			// Fall through: we still want to list files with parse errors.
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
			} else if fileListForExt((*Package)(p), ext) != nil {
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
			if list := fileListForExt((*Package)(p), ext); list != nil {
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
			badFile(name, &MultiplePackageError{
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
					badFile(name, fmt.Errorf("use of cgo in test %s not supported", name))
					continue
				}
				isCgo = true
			}
		}
		if directives := tf.cgoDirectives(); directives != "" {
			if err := ctxt.saveCgo(name, (*Package)(p), directives); err != nil {
				badFile(name, err)
			}
		}

		var fileList *[]string
		var importMap, embedMap map[string][]token.Position
		switch {
		case isCgo:
			allTags["cgo"] = true
			if ctxt.CgoEnabled {
				fileList = &p.CgoFiles
				importMap = importPos
				embedMap = embedPos
			} else {
				// Ignore Imports and Embeds from cgo files if cgo is disabled.
				fileList = &p.IgnoredGoFiles
			}
		case isXTest:
			fileList = &p.XTestGoFiles
			importMap = xTestImportPos
			embedMap = xTestEmbedPos
		case isTest:
			fileList = &p.TestGoFiles
			importMap = testImportPos
			embedMap = testEmbedPos
		default:
			fileList = &p.GoFiles
			importMap = importPos
			embedMap = embedPos
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
	mod, err := Get(modroot)
	if err != nil {
		return goroot.IsStandardPackage(goroot_, compiler, path)
	}

	pkgs := mod.Packages()
	i := sort.SearchStrings(pkgs, reldir)
	return i != len(pkgs) && pkgs[i] == reldir
}

// IsDirWithGoFiles is the equivalent of fsys.IsDirWithGoFiles using the information in the index.
func (mi *ModuleIndex) IsDirWithGoFiles(relpath string) (_ bool, err error) {
	rp := mi.indexPackage(relpath)

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
func (mi *ModuleIndex) ScanDir(path string, tags map[string]bool) (sortedImports []string, sortedTestImports []string, err error) {
	rp := mi.indexPackage(path)

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
			if imports.Eval(x, tags, true) == false {
				return false
			}
		}
	}

	return true
}

// index package holds the information needed to access information in the
// index about a package.
type indexPackage struct {
	error error
	dir   string // directory of the package relative to the modroot

	// Source files
	sourceFiles []*sourceFile
}

var errCannotFindPackage = errors.New("cannot find package")

// indexPackage returns an indexPackage constructed using the information in the ModuleIndex.
func (mi *ModuleIndex) indexPackage(path string) *indexPackage {
	defer func() {
		if e := recover(); e != nil {
			base.Fatalf("error reading module index: %v", e)
		}
	}()
	offset, ok := mi.packages[path]
	if !ok {
		return &indexPackage{error: fmt.Errorf("%w %q in:\n\t%s", errCannotFindPackage, path, filepath.Join(mi.modroot, path))}
	}

	// TODO(matloob): do we want to lock on the module index?
	d := mi.od.decoderAt(offset)
	rp := new(indexPackage)
	if errstr := d.string(); errstr != "" {
		rp.error = errors.New(errstr)
	}
	rp.dir = d.string()
	numSourceFiles := d.uint32()
	rp.sourceFiles = make([]*sourceFile, numSourceFiles)
	for i := uint32(0); i < numSourceFiles; i++ {
		offset := d.uint32()
		rp.sourceFiles[i] = &sourceFile{
			od: mi.od.offsetDecoderAt(offset),
		}
	}
	return rp
}

// sourceFile represents the information of a given source file in the module index.
type sourceFile struct {
	od offsetDecoder // od interprets all offsets relative to the start of the source file's data

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
	return sf.od.stringAt(sourceFileError)
}
func (sf *sourceFile) parseError() string {
	return sf.od.stringAt(sourceFileParseError)
}
func (sf *sourceFile) name() string {
	return sf.od.stringAt(sourceFileName)
}
func (sf *sourceFile) synopsis() string {
	return sf.od.stringAt(sourceFileSynopsis)
}
func (sf *sourceFile) pkgName() string {
	return sf.od.stringAt(sourceFilePkgName)
}
func (sf *sourceFile) ignoreFile() bool {
	return sf.od.boolAt(sourceFileIgnoreFile)
}
func (sf *sourceFile) binaryOnly() bool {
	return sf.od.boolAt(sourceFileBinaryOnly)
}
func (sf *sourceFile) cgoDirectives() string {
	return sf.od.stringAt(sourceFileCgoDirectives)
}
func (sf *sourceFile) goBuildConstraint() string {
	return sf.od.stringAt(sourceFileGoBuildConstraint)
}

func (sf *sourceFile) plusBuildConstraints() []string {
	d := sf.od.decoderAt(sourceFileNumPlusBuildConstraints)
	n := d.int()
	ret := make([]string, n)
	for i := 0; i < n; i++ {
		ret[i] = d.string()
	}
	return ret
}

func importsOffset(numPlusBuildConstraints int) int {
	// 4 bytes per uin32, add one to advance past numPlusBuildConstraints itself
	return sourceFileNumPlusBuildConstraints + 4*(numPlusBuildConstraints+1)
}

func (sf *sourceFile) importsOffset() int {
	numPlusBuildConstraints := sf.od.intAt(sourceFileNumPlusBuildConstraints)
	return importsOffset(numPlusBuildConstraints)
}

func embedsOffset(importsOffset, numImports int) int {
	// 4 bytes per uint32; 1 to advance past numImports itself, and 5 uint32s per import
	return importsOffset + 4*(1+(5*numImports))
}

func (sf *sourceFile) embedsOffset() int {
	importsOffset := sf.importsOffset()
	numImports := sf.od.intAt(importsOffset)
	return embedsOffset(importsOffset, numImports)
}

func (sf *sourceFile) imports() []rawImport {
	sf.onceReadImports.Do(func() {
		importsOffset := sf.importsOffset()
		d := sf.od.decoderAt(importsOffset)
		numImports := d.int()
		ret := make([]rawImport, numImports)
		for i := 0; i < numImports; i++ {
			ret[i].path = d.string()
			ret[i].position = d.tokpos()
		}
		sf.savedImports = ret
	})
	return sf.savedImports
}

func (sf *sourceFile) embeds() []embed {
	embedsOffset := sf.embedsOffset()
	d := sf.od.decoderAt(embedsOffset)
	numEmbeds := d.int()
	ret := make([]embed, numEmbeds)
	for i := range ret {
		pattern := d.string()
		pos := d.tokpos()
		ret[i] = embed{pattern, pos}
	}
	return ret
}

// A decoder reads from the current position of the file and advances its position as it
// reads.
type decoder struct {
	b  []byte
	st *stringTable
}

func (d *decoder) uint32() uint32 {
	n := binary.LittleEndian.Uint32(d.b[:4])
	d.b = d.b[4:]
	return n
}

func (d *decoder) int() int {
	n := d.uint32()
	if int64(n) > math.MaxInt {
		base.Fatalf("go: attempting to read a uint32 from the index that overflows int")
	}
	return int(n)
}

func (d *decoder) tokpos() token.Position {
	file := d.string()
	offset := d.int()
	line := d.int()
	column := d.int()
	return token.Position{
		Filename: file,
		Offset:   offset,
		Line:     line,
		Column:   column,
	}
}

func (d *decoder) string() string {
	return d.st.string(d.int())
}

// And offset decoder reads information offset from its position in the file.
// It's either offset from the beginning of the index, or the beginning of a sourceFile's data.
type offsetDecoder struct {
	b  []byte
	st *stringTable
}

func (od *offsetDecoder) uint32At(offset int) uint32 {
	if offset > len(od.b) {
		base.Fatalf("go: trying to read from index file at offset higher than file length. This indicates a corrupt offset file in the cache.")
	}
	return binary.LittleEndian.Uint32(od.b[offset:])
}

func (od *offsetDecoder) intAt(offset int) int {
	n := od.uint32At(offset)
	if int64(n) > math.MaxInt {
		base.Fatalf("go: attempting to read a uint32 from the index that overflows int")
	}
	return int(n)
}

func (od *offsetDecoder) boolAt(offset int) bool {
	switch v := od.uint32At(offset); v {
	case 0:
		return false
	case 1:
		return true
	default:
		base.Fatalf("go: invalid bool value in index file encoding: %v", v)
	}
	panic("unreachable")
}

func (od *offsetDecoder) stringAt(offset int) string {
	return od.st.string(od.intAt(offset))
}

func (od *offsetDecoder) decoderAt(offset int) *decoder {
	return &decoder{od.b[offset:], od.st}
}

func (od *offsetDecoder) offsetDecoderAt(offset uint32) offsetDecoder {
	return offsetDecoder{od.b[offset:], od.st}
}

type stringTable struct {
	b []byte
}

func newStringTable(b []byte) *stringTable {
	return &stringTable{b: b}
}

func (st *stringTable) string(pos int) string {
	if pos == 0 {
		return ""
	}

	bb := st.b[pos:]
	i := bytes.IndexByte(bb, 0)

	if i == -1 {
		panic("reached end of string table trying to read string")
	}
	s := asString(bb[:i])

	return s
}

func asString(b []byte) string {
	p := (*unsafeheader.Slice)(unsafe.Pointer(&b)).Data

	var s string
	hdr := (*unsafeheader.String)(unsafe.Pointer(&s))
	hdr.Data = p
	hdr.Len = len(b)

	return s
}
