// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"context"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/gopathwalk"
)

// Debug controls verbose logging.
var Debug = false

// LocalPrefix is a comma-separated string of import path prefixes, which, if
// set, instructs Process to sort the import paths with the given prefixes
// into another group after 3rd-party packages.
var LocalPrefix string

func localPrefixes() []string {
	if LocalPrefix != "" {
		return strings.Split(LocalPrefix, ",")
	}
	return nil
}

// importToGroup is a list of functions which map from an import path to
// a group number.
var importToGroup = []func(importPath string) (num int, ok bool){
	func(importPath string) (num int, ok bool) {
		for _, p := range localPrefixes() {
			if strings.HasPrefix(importPath, p) || strings.TrimSuffix(p, "/") == importPath {
				return 3, true
			}
		}
		return
	},
	func(importPath string) (num int, ok bool) {
		if strings.HasPrefix(importPath, "appengine") {
			return 2, true
		}
		return
	},
	func(importPath string) (num int, ok bool) {
		if strings.Contains(importPath, ".") {
			return 1, true
		}
		return
	},
}

func importGroup(importPath string) int {
	for _, fn := range importToGroup {
		if n, ok := fn(importPath); ok {
			return n
		}
	}
	return 0
}

// An importInfo represents a single import statement.
type importInfo struct {
	importPath string // import path, e.g. "crypto/rand".
	name       string // import name, e.g. "crand", or "" if none.
}

// A packageInfo represents what's known about a package.
type packageInfo struct {
	name    string          // discovered package name.
	exports map[string]bool // known exports.
}

// parseOtherFiles parses all the Go files in srcDir except filename, including
// test files if filename looks like a test.
func parseOtherFiles(fset *token.FileSet, srcDir, filename string) []*ast.File {
	considerTests := strings.HasSuffix(filename, "_test.go")

	fileBase := filepath.Base(filename)
	packageFileInfos, err := ioutil.ReadDir(srcDir)
	if err != nil {
		return nil
	}

	var files []*ast.File
	for _, fi := range packageFileInfos {
		if fi.Name() == fileBase || !strings.HasSuffix(fi.Name(), ".go") {
			continue
		}
		if !considerTests && strings.HasSuffix(fi.Name(), "_test.go") {
			continue
		}

		f, err := parser.ParseFile(fset, filepath.Join(srcDir, fi.Name()), nil, 0)
		if err != nil {
			continue
		}

		files = append(files, f)
	}

	return files
}

// addGlobals puts the names of package vars into the provided map.
func addGlobals(f *ast.File, globals map[string]bool) {
	for _, decl := range f.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}

		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			globals[valueSpec.Names[0].Name] = true
		}
	}
}

// collectReferences builds a map of selector expressions, from
// left hand side (X) to a set of right hand sides (Sel).
func collectReferences(f *ast.File) map[string]map[string]bool {
	refs := map[string]map[string]bool{}

	var visitor visitFn
	visitor = func(node ast.Node) ast.Visitor {
		if node == nil {
			return visitor
		}
		switch v := node.(type) {
		case *ast.SelectorExpr:
			xident, ok := v.X.(*ast.Ident)
			if !ok {
				break
			}
			if xident.Obj != nil {
				// if the parser can resolve it, it's not a package ref
				break
			}
			pkgName := xident.Name
			r := refs[pkgName]
			if r == nil {
				r = make(map[string]bool)
				refs[pkgName] = r
			}
			if ast.IsExported(v.Sel.Name) {
				r[v.Sel.Name] = true
			}
		}
		return visitor
	}
	ast.Walk(visitor, f)
	return refs
}

// collectImports returns all the imports in f, keyed by their package name as
// determined by pathToName. Unnamed imports (., _) and "C" are ignored.
func collectImports(f *ast.File) []*importInfo {
	var imports []*importInfo
	for _, imp := range f.Imports {
		var name string
		if imp.Name != nil {
			name = imp.Name.Name
		}
		if imp.Path.Value == `"C"` || name == "_" || name == "." {
			continue
		}
		path := strings.Trim(imp.Path.Value, `"`)
		imports = append(imports, &importInfo{
			name:       name,
			importPath: path,
		})
	}
	return imports
}

// findMissingImport searches pass's candidates for an import that provides
// pkg, containing all of syms.
func (p *pass) findMissingImport(pkg string, syms map[string]bool) *importInfo {
	for _, candidate := range p.candidates {
		pkgInfo, ok := p.knownPackages[candidate.importPath]
		if !ok {
			continue
		}
		if candidate.name != "" && candidate.name != pkg {
			continue
		}
		if pkgInfo.name != pkg {
			continue
		}

		allFound := true
		for right := range syms {
			if !pkgInfo.exports[right] {
				allFound = false
				break
			}
		}

		if allFound {
			return candidate
		}
	}
	return nil
}

// A pass contains all the inputs and state necessary to fix a file's imports.
// It can be modified in some ways during use; see comments below.
type pass struct {
	// Inputs. These must be set before a call to load, and not modified after.
	fset       *token.FileSet                          // fset used to parse f and its siblings.
	f          *ast.File                               // the file being fixed.
	otherFiles []*ast.File                             // sibling files.
	srcDir     string                                  // the directory containing f.
	pathToName func(path string, srcDir string) string // the function to use to resolve an import path to a package name.

	// Intermediate state, generated by load.
	existingImports map[string]*importInfo
	missing         map[string]map[string]bool
	used            map[*importInfo]bool

	// Inputs to fix. These can be augmented between successive fix calls.
	lastTry        bool                    // indicates that this is the last call and fix should clean up as best it can.
	addImportNames bool                    // add names to mismatched imports.
	candidates     []*importInfo           // candidate imports in priority order.
	knownPackages  map[string]*packageInfo // information about all known packages.
}

// loadPackageNames saves the package names for everything referenced by imports.
func (p *pass) loadPackageNames(imports []*importInfo) {
	for _, imp := range imports {
		if _, ok := p.knownPackages[imp.importPath]; ok {
			continue
		}
		p.knownPackages[imp.importPath] = &packageInfo{
			name:    p.pathToName(imp.importPath, p.srcDir),
			exports: map[string]bool{},
		}
	}
}

// importIdentifier returns the indentifier that imp will introduce.
func (p *pass) importIdentifier(imp *importInfo) string {
	if imp.name != "" {
		return imp.name
	}
	return p.knownPackages[imp.importPath].name
}

// load reads in everything necessary to run a pass, and reports whether the
// file already has all the imports it needs. It fills in p.missing with the
// file's missing symbols, if any, or removes unused imports if not.
func (p *pass) load() bool {
	p.knownPackages = map[string]*packageInfo{}
	p.missing = map[string]map[string]bool{}
	p.used = map[*importInfo]bool{}
	p.existingImports = map[string]*importInfo{}

	// Load basic information about the file in question.
	allReferences := collectReferences(p.f)

	// Load stuff from other files in the same package:
	// global variables so we know they don't need resolving, and imports
	// that we might want to mimic.
	globals := map[string]bool{}
	for _, otherFile := range p.otherFiles {
		addGlobals(otherFile, globals)
		p.candidates = append(p.candidates, collectImports(otherFile)...)
	}

	// Resolve all the import paths we've seen to package names, and store
	// f's imports by the identifier they introduce.
	imports := collectImports(p.f)
	p.loadPackageNames(append(imports, p.candidates...))
	for _, imp := range imports {
		p.existingImports[p.importIdentifier(imp)] = imp
	}

	// Find missing references and mark used imports used.
	for left, rights := range allReferences {
		if globals[left] {
			continue
		}
		imp, ok := p.existingImports[left]
		if !ok {
			p.missing[left] = rights
			continue
		}
		p.used[imp] = true
	}
	if len(p.missing) != 0 {
		return false
	}

	return p.fix()
}

// fix attempts to satisfy missing imports using p.candidates. If it finds
// everything, or if p.lastTry is true, it adds the imports it found,
// removes anything unused, and returns true.
func (p *pass) fix() bool {
	// Find missing imports.
	var selected []*importInfo
	for left, rights := range p.missing {
		if imp := p.findMissingImport(left, rights); imp != nil {
			selected = append(selected, imp)
		}
	}

	if !p.lastTry && len(selected) != len(p.missing) {
		return false
	}

	// Found everything, or giving up. Add the new imports and remove any unused.
	for _, imp := range p.existingImports {
		if !p.used[imp] {
			astutil.DeleteNamedImport(p.fset, p.f, imp.name, imp.importPath)
		}
	}

	if p.addImportNames {
		for _, imp := range p.f.Imports {
			if imp.Name != nil {
				continue
			}
			path := strings.Trim(imp.Path.Value, `""`)
			pkg, ok := p.knownPackages[path]
			if !ok {
				continue
			}
			if pkg.name != importPathToNameBasic(path, p.srcDir) {
				imp.Name = &ast.Ident{Name: pkg.name, NamePos: imp.Pos()}
			}
		}
	}

	for _, imp := range selected {
		astutil.AddNamedImport(p.fset, p.f, imp.name, imp.importPath)
	}

	return true
}

// assumeSiblingImportsValid assumes that siblings' use of packages is valid,
// adding the exports they use.
func (p *pass) assumeSiblingImportsValid() {
	for _, f := range p.otherFiles {
		refs := collectReferences(f)
		imports := collectImports(f)
		importsByName := map[string]*importInfo{}
		for _, imp := range imports {
			importsByName[p.importIdentifier(imp)] = imp
		}
		for left, rights := range refs {
			if imp, ok := importsByName[left]; ok {
				if _, ok := stdlib[imp.importPath]; ok {
					// We have the stdlib in memory; no need to guess.
					continue
				}
				p.addCandidate(imp, &packageInfo{
					// no name; we already know it.
					exports: rights,
				})
			}
		}
	}
}

// addCandidate adds a candidate import to p, and merges in the information
// in pkg.
func (p *pass) addCandidate(imp *importInfo, pkg *packageInfo) {
	p.candidates = append(p.candidates, imp)
	if existing, ok := p.knownPackages[imp.importPath]; ok {
		for export := range pkg.exports {
			existing.exports[export] = true
		}
	} else {
		p.knownPackages[imp.importPath] = pkg
	}
}

func fixImports(fset *token.FileSet, f *ast.File, filename string) error {
	abs, err := filepath.Abs(filename)
	if err != nil {
		return err
	}
	srcDir := filepath.Dir(abs)
	if Debug {
		log.Printf("fixImports(filename=%q), abs=%q, srcDir=%q ...", filename, abs, srcDir)
	}

	// First pass: looking only at f, and using the naive algorithm to
	// derive package names from import paths, see if the file is already
	// complete. We can't add any imports yet, because we don't know
	// if missing references are actually package vars.
	p := &pass{fset: fset, f: f, srcDir: srcDir}
	p.pathToName = importPathToNameBasic
	if p.load() {
		return nil
	}

	otherFiles := parseOtherFiles(fset, srcDir, filename)

	// Second pass: add information from other files in the same package,
	// like their package vars and imports.
	p = &pass{fset: fset, f: f, srcDir: srcDir}
	p.pathToName = importPathToNameBasic
	p.otherFiles = otherFiles
	if p.load() {
		return nil
	}

	// Now we can try adding imports from the stdlib.
	p.assumeSiblingImportsValid()
	addStdlibCandidates(p, p.missing)
	if p.fix() {
		return nil
	}

	// Third pass: get real package names where we had previously used
	// the naive algorithm.
	p = &pass{fset: fset, f: f, srcDir: srcDir}
	p.pathToName = importPathToName
	p.addImportNames = true
	p.otherFiles = otherFiles
	if p.load() {
		return nil
	}

	addStdlibCandidates(p, p.missing)
	p.assumeSiblingImportsValid()
	if p.fix() {
		return nil
	}

	// Go look for candidates in $GOPATH, etc. We don't necessarily load
	// the real exports of sibling imports, so keep assuming their contents.
	if err := addGoPathCandidates(p, p.missing, filename); err != nil {
		return err
	}

	p.lastTry = true
	p.fix()
	return nil
}

func addStdlibCandidates(pass *pass, refs map[string]map[string]bool) {
	add := func(pkg string) {
		pass.addCandidate(
			&importInfo{importPath: pkg},
			&packageInfo{name: path.Base(pkg), exports: stdlib[pkg]})
	}
	for left := range refs {
		if left == "rand" {
			// Make sure we try crypto/rand before math/rand.
			add("crypto/rand")
			add("math/rand")
			continue
		}
		for importPath := range stdlib {
			if path.Base(importPath) == left {
				add(importPath)
			}
		}
	}
}

func addGoPathCandidates(pass *pass, refs map[string]map[string]bool, filename string) error {
	// Search for imports matching potential package references.
	type result struct {
		imp *importInfo
		pkg *packageInfo
	}
	results := make(chan result, len(refs))

	ctx, cancel := context.WithCancel(context.TODO())
	var wg sync.WaitGroup
	defer func() {
		cancel()
		wg.Wait()
	}()
	var (
		firstErr     error
		firstErrOnce sync.Once
	)
	for pkgName, symbols := range refs {
		wg.Add(1)
		go func(pkgName string, symbols map[string]bool) {
			defer wg.Done()

			ipath, _, err := findImport(ctx, pkgName, symbols, filename)

			if err != nil {
				firstErrOnce.Do(func() {
					firstErr = err
					cancel()
				})
				return
			}

			if ipath == "" {
				return // No matching package.
			}

			imp := &importInfo{
				importPath: ipath,
			}
			// If the package name isn't what you'd expect looking
			// at the import path, add an explicit name.
			if importPathToNameBasic(ipath, pass.srcDir) != pkgName {
				imp.name = pkgName
			}

			pkg := &packageInfo{
				name:    pkgName,
				exports: symbols,
			}
			results <- result{imp, pkg}
		}(pkgName, symbols)
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	for result := range results {
		pass.addCandidate(result.imp, result.pkg)
	}
	return firstErr
}

// importPathToName returns the package name for the given import path.
var importPathToName func(importPath, srcDir string) (packageName string) = importPathToNameGoPath

// importPathToNameBasic assumes the package name is the base of import path,
// except that if the path ends in foo/vN, it assumes the package name is foo.
func importPathToNameBasic(importPath, srcDir string) (packageName string) {
	base := path.Base(importPath)
	if strings.HasPrefix(base, "v") {
		if _, err := strconv.Atoi(base[1:]); err == nil {
			dir := path.Dir(importPath)
			if dir != "." {
				return path.Base(dir)
			}
		}
	}
	return base
}

// importPathToNameGoPath finds out the actual package name, as declared in its .go files.
// If there's a problem, it falls back to using importPathToNameBasic.
func importPathToNameGoPath(importPath, srcDir string) (packageName string) {
	// Fast path for standard library without going to disk.
	if _, ok := stdlib[importPath]; ok {
		return path.Base(importPath) // stdlib packages always match their paths.
	}

	pkgName, err := importPathToNameGoPathParse(importPath, srcDir)
	if Debug {
		log.Printf("importPathToNameGoPathParse(%q, srcDir=%q) = %q, %v", importPath, srcDir, pkgName, err)
	}
	if err == nil {
		return pkgName
	}
	return importPathToNameBasic(importPath, srcDir)
}

// importPathToNameGoPathParse is a faster version of build.Import if
// the only thing desired is the package name. It uses build.FindOnly
// to find the directory and then only parses one file in the package,
// trusting that the files in the directory are consistent.
func importPathToNameGoPathParse(importPath, srcDir string) (packageName string, err error) {
	buildPkg, err := build.Import(importPath, srcDir, build.FindOnly)
	if err != nil {
		return "", err
	}
	d, err := os.Open(buildPkg.Dir)
	if err != nil {
		return "", err
	}
	names, err := d.Readdirnames(-1)
	d.Close()
	if err != nil {
		return "", err
	}
	sort.Strings(names) // to have predictable behavior
	var lastErr error
	var nfile int
	for _, name := range names {
		if !strings.HasSuffix(name, ".go") {
			continue
		}
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		nfile++
		fullFile := filepath.Join(buildPkg.Dir, name)

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, fullFile, nil, parser.PackageClauseOnly)
		if err != nil {
			lastErr = err
			continue
		}
		pkgName := f.Name.Name
		if pkgName == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by ctx.MatchFile.
			continue
		}
		if pkgName == "main" {
			// Also skip package main, assuming it's a +build ignore generator or example.
			// Since you can't import a package main anyway, there's no harm here.
			continue
		}
		return pkgName, nil
	}
	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("no importable package found in %d Go files", nfile)
}

// Directory-scanning state.
var (
	// scanOnce guards calling scanGoDirs and assigning dirScan
	scanOnce sync.Once
	dirScan  map[string]*pkg // abs dir path => *pkg
)

type pkg struct {
	dir             string // absolute file path to pkg directory ("/usr/lib/go/src/net/http")
	importPath      string // full pkg import path ("net/http", "foo/bar/vendor/a/b")
	importPathShort string // vendorless import path ("net/http", "a/b")
}

type pkgDistance struct {
	pkg      *pkg
	distance int // relative distance to target
}

// byDistanceOrImportPathShortLength sorts by relative distance breaking ties
// on the short import path length and then the import string itself.
type byDistanceOrImportPathShortLength []pkgDistance

func (s byDistanceOrImportPathShortLength) Len() int { return len(s) }
func (s byDistanceOrImportPathShortLength) Less(i, j int) bool {
	di, dj := s[i].distance, s[j].distance
	if di == -1 {
		return false
	}
	if dj == -1 {
		return true
	}
	if di != dj {
		return di < dj
	}

	vi, vj := s[i].pkg.importPathShort, s[j].pkg.importPathShort
	if len(vi) != len(vj) {
		return len(vi) < len(vj)
	}
	return vi < vj
}
func (s byDistanceOrImportPathShortLength) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func distance(basepath, targetpath string) int {
	p, err := filepath.Rel(basepath, targetpath)
	if err != nil {
		return -1
	}
	if p == "." {
		return 0
	}
	return strings.Count(p, string(filepath.Separator)) + 1
}

// scanGoDirs populates the dirScan map for GOPATH and GOROOT.
func scanGoDirs() map[string]*pkg {
	result := make(map[string]*pkg)
	var mu sync.Mutex

	add := func(root gopathwalk.Root, dir string) {
		mu.Lock()
		defer mu.Unlock()

		if _, dup := result[dir]; dup {
			return
		}
		importpath := filepath.ToSlash(dir[len(root.Path)+len("/"):])
		result[dir] = &pkg{
			importPath:      importpath,
			importPathShort: VendorlessPath(importpath),
			dir:             dir,
		}
	}
	gopathwalk.Walk(gopathwalk.SrcDirsRoots(), add, gopathwalk.Options{Debug: Debug, ModulesEnabled: false})
	return result
}

// VendorlessPath returns the devendorized version of the import path ipath.
// For example, VendorlessPath("foo/bar/vendor/a/b") returns "a/b".
func VendorlessPath(ipath string) string {
	// Devendorize for use in import statement.
	if i := strings.LastIndex(ipath, "/vendor/"); i >= 0 {
		return ipath[i+len("/vendor/"):]
	}
	if strings.HasPrefix(ipath, "vendor/") {
		return ipath[len("vendor/"):]
	}
	return ipath
}

// loadExports returns the set of exported symbols in the package at dir.
// It returns nil on error or if the package name in dir does not match expectPackage.
var loadExports func(ctx context.Context, expectPackage, dir string) (map[string]bool, error) = loadExportsGoPath

func loadExportsGoPath(ctx context.Context, expectPackage, dir string) (map[string]bool, error) {
	if Debug {
		log.Printf("loading exports in dir %s (seeking package %s)", dir, expectPackage)
	}
	exports := make(map[string]bool)

	buildCtx := build.Default

	// ReadDir is like ioutil.ReadDir, but only returns *.go files
	// and filters out _test.go files since they're not relevant
	// and only slow things down.
	buildCtx.ReadDir = func(dir string) (notTests []os.FileInfo, err error) {
		all, err := ioutil.ReadDir(dir)
		if err != nil {
			return nil, err
		}
		notTests = all[:0]
		for _, fi := range all {
			name := fi.Name()
			if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
				notTests = append(notTests, fi)
			}
		}
		return notTests, nil
	}

	files, err := buildCtx.ReadDir(dir)
	if err != nil {
		log.Print(err)
		return nil, err
	}

	fset := token.NewFileSet()

	for _, fi := range files {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		match, err := buildCtx.MatchFile(dir, fi.Name())
		if err != nil || !match {
			continue
		}
		fullFile := filepath.Join(dir, fi.Name())
		f, err := parser.ParseFile(fset, fullFile, nil, 0)
		if err != nil {
			if Debug {
				log.Printf("Parsing %s: %v", fullFile, err)
			}
			return nil, err
		}
		pkgName := f.Name.Name
		if pkgName == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by buildCtx.MatchFile.
			continue
		}
		if pkgName != expectPackage {
			err := fmt.Errorf("scan of dir %v is not expected package %v (actually %v)", dir, expectPackage, pkgName)
			if Debug {
				log.Print(err)
			}
			return nil, err
		}
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				exports[name] = true
			}
		}
	}

	if Debug {
		exportList := make([]string, 0, len(exports))
		for k := range exports {
			exportList = append(exportList, k)
		}
		sort.Strings(exportList)
		log.Printf("loaded exports in dir %v (package %v): %v", dir, expectPackage, strings.Join(exportList, ", "))
	}
	return exports, nil
}

// findImport searches for a package with the given symbols.
// If no package is found, findImport returns ("", false, nil)
//
// This is declared as a variable rather than a function so goimports
// can be easily extended by adding a file with an init function.
//
// The rename value is ignored.
var findImport func(ctx context.Context, pkgName string, symbols map[string]bool, filename string) (foundPkg string, rename bool, err error) = findImportGoPath

// findImportGoPath is the normal implementation of findImport.
// (Some companies have their own internally.)
func findImportGoPath(ctx context.Context, pkgName string, symbols map[string]bool, filename string) (foundPkg string, rename bool, err error) {
	pkgDir, err := filepath.Abs(filename)
	if err != nil {
		return "", false, err
	}
	pkgDir = filepath.Dir(pkgDir)
	// Scan $GOROOT and each $GOPATH.
	scanOnce.Do(func() { dirScan = scanGoDirs() })

	// Find candidate packages, looking only at their directory names first.
	var candidates []pkgDistance
	for _, pkg := range dirScan {
		if pkgIsCandidate(filename, pkgName, pkg) {
			candidates = append(candidates, pkgDistance{
				pkg:      pkg,
				distance: distance(pkgDir, pkg.dir),
			})
		}
	}

	// Sort the candidates by their import package length,
	// assuming that shorter package names are better than long
	// ones.  Note that this sorts by the de-vendored name, so
	// there's no "penalty" for vendoring.
	sort.Sort(byDistanceOrImportPathShortLength(candidates))
	if Debug {
		for i, c := range candidates {
			log.Printf("%s candidate %d/%d: %v in %v", pkgName, i+1, len(candidates), c.pkg.importPathShort, c.pkg.dir)
		}
	}

	// Collect exports for packages with matching names.

	rescv := make([]chan *pkg, len(candidates))
	for i := range candidates {
		rescv[i] = make(chan *pkg, 1)
	}
	const maxConcurrentPackageImport = 4
	loadExportsSem := make(chan struct{}, maxConcurrentPackageImport)

	ctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	defer func() {
		cancel()
		wg.Wait()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, c := range candidates {
			select {
			case loadExportsSem <- struct{}{}:
			case <-ctx.Done():
				return
			}

			wg.Add(1)
			go func(c pkgDistance, resc chan<- *pkg) {
				defer func() {
					<-loadExportsSem
					wg.Done()
				}()

				exports, err := loadExports(ctx, pkgName, c.pkg.dir)
				if err != nil {
					resc <- nil
					return
				}

				// If it doesn't have the right
				// symbols, send nil to mean no match.
				for symbol := range symbols {
					if !exports[symbol] {
						resc <- nil
						return
					}
				}
				resc <- c.pkg
			}(c, rescv[i])
		}
	}()

	for _, resc := range rescv {
		pkg := <-resc
		if pkg == nil {
			continue
		}
		return pkg.importPathShort, false, nil
	}
	return "", false, nil
}

// pkgIsCandidate reports whether pkg is a candidate for satisfying the
// finding which package pkgIdent in the file named by filename is trying
// to refer to.
//
// This check is purely lexical and is meant to be as fast as possible
// because it's run over all $GOPATH directories to filter out poor
// candidates in order to limit the CPU and I/O later parsing the
// exports in candidate packages.
//
// filename is the file being formatted.
// pkgIdent is the package being searched for, like "client" (if
// searching for "client.New")
func pkgIsCandidate(filename, pkgIdent string, pkg *pkg) bool {
	// Check "internal" and "vendor" visibility:
	if !canUse(filename, pkg.dir) {
		return false
	}

	// Speed optimization to minimize disk I/O:
	// the last two components on disk must contain the
	// package name somewhere.
	//
	// This permits mismatch naming like directory
	// "go-foo" being package "foo", or "pkg.v3" being "pkg",
	// or directory "google.golang.org/api/cloudbilling/v1"
	// being package "cloudbilling", but doesn't
	// permit a directory "foo" to be package
	// "bar", which is strongly discouraged
	// anyway. There's no reason goimports needs
	// to be slow just to accommodate that.
	lastTwo := lastTwoComponents(pkg.importPathShort)
	if strings.Contains(lastTwo, pkgIdent) {
		return true
	}
	if hasHyphenOrUpperASCII(lastTwo) && !hasHyphenOrUpperASCII(pkgIdent) {
		lastTwo = lowerASCIIAndRemoveHyphen(lastTwo)
		if strings.Contains(lastTwo, pkgIdent) {
			return true
		}
	}

	return false
}

func hasHyphenOrUpperASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b == '-' || ('A' <= b && b <= 'Z') {
			return true
		}
	}
	return false
}

func lowerASCIIAndRemoveHyphen(s string) (ret string) {
	buf := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		b := s[i]
		switch {
		case b == '-':
			continue
		case 'A' <= b && b <= 'Z':
			buf = append(buf, b+('a'-'A'))
		default:
			buf = append(buf, b)
		}
	}
	return string(buf)
}

// canUse reports whether the package in dir is usable from filename,
// respecting the Go "internal" and "vendor" visibility rules.
func canUse(filename, dir string) bool {
	// Fast path check, before any allocations. If it doesn't contain vendor
	// or internal, it's not tricky:
	// Note that this can false-negative on directories like "notinternal",
	// but we check it correctly below. This is just a fast path.
	if !strings.Contains(dir, "vendor") && !strings.Contains(dir, "internal") {
		return true
	}

	dirSlash := filepath.ToSlash(dir)
	if !strings.Contains(dirSlash, "/vendor/") && !strings.Contains(dirSlash, "/internal/") && !strings.HasSuffix(dirSlash, "/internal") {
		return true
	}
	// Vendor or internal directory only visible from children of parent.
	// That means the path from the current directory to the target directory
	// can contain ../vendor or ../internal but not ../foo/vendor or ../foo/internal
	// or bar/vendor or bar/internal.
	// After stripping all the leading ../, the only okay place to see vendor or internal
	// is at the very beginning of the path.
	absfile, err := filepath.Abs(filename)
	if err != nil {
		return false
	}
	absdir, err := filepath.Abs(dir)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(absfile, absdir)
	if err != nil {
		return false
	}
	relSlash := filepath.ToSlash(rel)
	if i := strings.LastIndex(relSlash, "../"); i >= 0 {
		relSlash = relSlash[i+len("../"):]
	}
	return !strings.Contains(relSlash, "/vendor/") && !strings.Contains(relSlash, "/internal/") && !strings.HasSuffix(relSlash, "/internal")
}

// lastTwoComponents returns at most the last two path components
// of v, using either / or \ as the path separator.
func lastTwoComponents(v string) string {
	nslash := 0
	for i := len(v) - 1; i >= 0; i-- {
		if v[i] == '/' || v[i] == '\\' {
			nslash++
			if nslash == 2 {
				return v[i:]
			}
		}
	}
	return v
}

type visitFn func(node ast.Node) ast.Visitor

func (fn visitFn) Visit(node ast.Node) ast.Visitor {
	return fn(node)
}
