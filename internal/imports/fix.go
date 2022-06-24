// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/gopathwalk"
)

// importToGroup is a list of functions which map from an import path to
// a group number.
var importToGroup = []func(localPrefix, importPath string) (num int, ok bool){
	func(localPrefix, importPath string) (num int, ok bool) {
		if localPrefix == "" {
			return
		}
		for _, p := range strings.Split(localPrefix, ",") {
			if strings.HasPrefix(importPath, p) || strings.TrimSuffix(p, "/") == importPath {
				return 3, true
			}
		}
		return
	},
	func(_, importPath string) (num int, ok bool) {
		if strings.HasPrefix(importPath, "appengine") {
			return 2, true
		}
		return
	},
	func(_, importPath string) (num int, ok bool) {
		firstComponent := strings.Split(importPath, "/")[0]
		if strings.Contains(firstComponent, ".") {
			return 1, true
		}
		return
	},
}

func importGroup(localPrefix, importPath string) int {
	for _, fn := range importToGroup {
		if n, ok := fn(localPrefix, importPath); ok {
			return n
		}
	}
	return 0
}

type ImportFixType int

const (
	AddImport ImportFixType = iota
	DeleteImport
	SetImportName
)

type ImportFix struct {
	// StmtInfo represents the import statement this fix will add, remove, or change.
	StmtInfo ImportInfo
	// IdentName is the identifier that this fix will add or remove.
	IdentName string
	// FixType is the type of fix this is (AddImport, DeleteImport, SetImportName).
	FixType   ImportFixType
	Relevance float64 // see pkg
}

// An ImportInfo represents a single import statement.
type ImportInfo struct {
	ImportPath string // import path, e.g. "crypto/rand".
	Name       string // import name, e.g. "crand", or "" if none.
}

// A packageInfo represents what's known about a package.
type packageInfo struct {
	name    string          // real package name, if known.
	exports map[string]bool // known exports.
}

// parseOtherFiles parses all the Go files in srcDir except filename, including
// test files if filename looks like a test.
func parseOtherFiles(fset *token.FileSet, srcDir, filename string) []*ast.File {
	// This could use go/packages but it doesn't buy much, and it fails
	// with https://golang.org/issue/26296 in LoadFiles mode in some cases.
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
func collectReferences(f *ast.File) references {
	refs := references{}

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
				// If the parser can resolve it, it's not a package ref.
				break
			}
			if !ast.IsExported(v.Sel.Name) {
				// Whatever this is, it's not exported from a package.
				break
			}
			pkgName := xident.Name
			r := refs[pkgName]
			if r == nil {
				r = make(map[string]bool)
				refs[pkgName] = r
			}
			r[v.Sel.Name] = true
		}
		return visitor
	}
	ast.Walk(visitor, f)
	return refs
}

// collectImports returns all the imports in f.
// Unnamed imports (., _) and "C" are ignored.
func collectImports(f *ast.File) []*ImportInfo {
	var imports []*ImportInfo
	for _, imp := range f.Imports {
		var name string
		if imp.Name != nil {
			name = imp.Name.Name
		}
		if imp.Path.Value == `"C"` || name == "_" || name == "." {
			continue
		}
		path := strings.Trim(imp.Path.Value, `"`)
		imports = append(imports, &ImportInfo{
			Name:       name,
			ImportPath: path,
		})
	}
	return imports
}

// findMissingImport searches pass's candidates for an import that provides
// pkg, containing all of syms.
func (p *pass) findMissingImport(pkg string, syms map[string]bool) *ImportInfo {
	for _, candidate := range p.candidates {
		pkgInfo, ok := p.knownPackages[candidate.ImportPath]
		if !ok {
			continue
		}
		if p.importIdentifier(candidate) != pkg {
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

// references is set of references found in a Go file. The first map key is the
// left hand side of a selector expression, the second key is the right hand
// side, and the value should always be true.
type references map[string]map[string]bool

// A pass contains all the inputs and state necessary to fix a file's imports.
// It can be modified in some ways during use; see comments below.
type pass struct {
	// Inputs. These must be set before a call to load, and not modified after.
	fset                 *token.FileSet // fset used to parse f and its siblings.
	f                    *ast.File      // the file being fixed.
	srcDir               string         // the directory containing f.
	env                  *ProcessEnv    // the environment to use for go commands, etc.
	loadRealPackageNames bool           // if true, load package names from disk rather than guessing them.
	otherFiles           []*ast.File    // sibling files.

	// Intermediate state, generated by load.
	existingImports map[string]*ImportInfo
	allRefs         references
	missingRefs     references

	// Inputs to fix. These can be augmented between successive fix calls.
	lastTry       bool                    // indicates that this is the last call and fix should clean up as best it can.
	candidates    []*ImportInfo           // candidate imports in priority order.
	knownPackages map[string]*packageInfo // information about all known packages.
}

// loadPackageNames saves the package names for everything referenced by imports.
func (p *pass) loadPackageNames(imports []*ImportInfo) error {
	if p.env.Logf != nil {
		p.env.Logf("loading package names for %v packages", len(imports))
		defer func() {
			p.env.Logf("done loading package names for %v packages", len(imports))
		}()
	}
	var unknown []string
	for _, imp := range imports {
		if _, ok := p.knownPackages[imp.ImportPath]; ok {
			continue
		}
		unknown = append(unknown, imp.ImportPath)
	}

	resolver, err := p.env.GetResolver()
	if err != nil {
		return err
	}

	names, err := resolver.loadPackageNames(unknown, p.srcDir)
	if err != nil {
		return err
	}

	for path, name := range names {
		p.knownPackages[path] = &packageInfo{
			name:    name,
			exports: map[string]bool{},
		}
	}
	return nil
}

// importIdentifier returns the identifier that imp will introduce. It will
// guess if the package name has not been loaded, e.g. because the source
// is not available.
func (p *pass) importIdentifier(imp *ImportInfo) string {
	if imp.Name != "" {
		return imp.Name
	}
	known := p.knownPackages[imp.ImportPath]
	if known != nil && known.name != "" {
		return known.name
	}
	return ImportPathToAssumedName(imp.ImportPath)
}

// load reads in everything necessary to run a pass, and reports whether the
// file already has all the imports it needs. It fills in p.missingRefs with the
// file's missing symbols, if any, or removes unused imports if not.
func (p *pass) load() ([]*ImportFix, bool) {
	p.knownPackages = map[string]*packageInfo{}
	p.missingRefs = references{}
	p.existingImports = map[string]*ImportInfo{}

	// Load basic information about the file in question.
	p.allRefs = collectReferences(p.f)

	// Load stuff from other files in the same package:
	// global variables so we know they don't need resolving, and imports
	// that we might want to mimic.
	globals := map[string]bool{}
	for _, otherFile := range p.otherFiles {
		// Don't load globals from files that are in the same directory
		// but a different package. Using them to suggest imports is OK.
		if p.f.Name.Name == otherFile.Name.Name {
			addGlobals(otherFile, globals)
		}
		p.candidates = append(p.candidates, collectImports(otherFile)...)
	}

	// Resolve all the import paths we've seen to package names, and store
	// f's imports by the identifier they introduce.
	imports := collectImports(p.f)
	if p.loadRealPackageNames {
		err := p.loadPackageNames(append(imports, p.candidates...))
		if err != nil {
			if p.env.Logf != nil {
				p.env.Logf("loading package names: %v", err)
			}
			return nil, false
		}
	}
	for _, imp := range imports {
		p.existingImports[p.importIdentifier(imp)] = imp
	}

	// Find missing references.
	for left, rights := range p.allRefs {
		if globals[left] {
			continue
		}
		_, ok := p.existingImports[left]
		if !ok {
			p.missingRefs[left] = rights
			continue
		}
	}
	if len(p.missingRefs) != 0 {
		return nil, false
	}

	return p.fix()
}

// fix attempts to satisfy missing imports using p.candidates. If it finds
// everything, or if p.lastTry is true, it updates fixes to add the imports it found,
// delete anything unused, and update import names, and returns true.
func (p *pass) fix() ([]*ImportFix, bool) {
	// Find missing imports.
	var selected []*ImportInfo
	for left, rights := range p.missingRefs {
		if imp := p.findMissingImport(left, rights); imp != nil {
			selected = append(selected, imp)
		}
	}

	if !p.lastTry && len(selected) != len(p.missingRefs) {
		return nil, false
	}

	// Found everything, or giving up. Add the new imports and remove any unused.
	var fixes []*ImportFix
	for _, imp := range p.existingImports {
		// We deliberately ignore globals here, because we can't be sure
		// they're in the same package. People do things like put multiple
		// main packages in the same directory, and we don't want to
		// remove imports if they happen to have the same name as a var in
		// a different package.
		if _, ok := p.allRefs[p.importIdentifier(imp)]; !ok {
			fixes = append(fixes, &ImportFix{
				StmtInfo:  *imp,
				IdentName: p.importIdentifier(imp),
				FixType:   DeleteImport,
			})
			continue
		}

		// An existing import may need to update its import name to be correct.
		if name := p.importSpecName(imp); name != imp.Name {
			fixes = append(fixes, &ImportFix{
				StmtInfo: ImportInfo{
					Name:       name,
					ImportPath: imp.ImportPath,
				},
				IdentName: p.importIdentifier(imp),
				FixType:   SetImportName,
			})
		}
	}

	for _, imp := range selected {
		fixes = append(fixes, &ImportFix{
			StmtInfo: ImportInfo{
				Name:       p.importSpecName(imp),
				ImportPath: imp.ImportPath,
			},
			IdentName: p.importIdentifier(imp),
			FixType:   AddImport,
		})
	}

	return fixes, true
}

// importSpecName gets the import name of imp in the import spec.
//
// When the import identifier matches the assumed import name, the import name does
// not appear in the import spec.
func (p *pass) importSpecName(imp *ImportInfo) string {
	// If we did not load the real package names, or the name is already set,
	// we just return the existing name.
	if !p.loadRealPackageNames || imp.Name != "" {
		return imp.Name
	}

	ident := p.importIdentifier(imp)
	if ident == ImportPathToAssumedName(imp.ImportPath) {
		return "" // ident not needed since the assumed and real names are the same.
	}
	return ident
}

// apply will perform the fixes on f in order.
func apply(fset *token.FileSet, f *ast.File, fixes []*ImportFix) {
	for _, fix := range fixes {
		switch fix.FixType {
		case DeleteImport:
			astutil.DeleteNamedImport(fset, f, fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
		case AddImport:
			astutil.AddNamedImport(fset, f, fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
		case SetImportName:
			// Find the matching import path and change the name.
			for _, spec := range f.Imports {
				path := strings.Trim(spec.Path.Value, `"`)
				if path == fix.StmtInfo.ImportPath {
					spec.Name = &ast.Ident{
						Name:    fix.StmtInfo.Name,
						NamePos: spec.Pos(),
					}
				}
			}
		}
	}
}

// assumeSiblingImportsValid assumes that siblings' use of packages is valid,
// adding the exports they use.
func (p *pass) assumeSiblingImportsValid() {
	for _, f := range p.otherFiles {
		refs := collectReferences(f)
		imports := collectImports(f)
		importsByName := map[string]*ImportInfo{}
		for _, imp := range imports {
			importsByName[p.importIdentifier(imp)] = imp
		}
		for left, rights := range refs {
			if imp, ok := importsByName[left]; ok {
				if m, ok := stdlib[imp.ImportPath]; ok {
					// We have the stdlib in memory; no need to guess.
					rights = copyExports(m)
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
func (p *pass) addCandidate(imp *ImportInfo, pkg *packageInfo) {
	p.candidates = append(p.candidates, imp)
	if existing, ok := p.knownPackages[imp.ImportPath]; ok {
		if existing.name == "" {
			existing.name = pkg.name
		}
		for export := range pkg.exports {
			existing.exports[export] = true
		}
	} else {
		p.knownPackages[imp.ImportPath] = pkg
	}
}

// fixImports adds and removes imports from f so that all its references are
// satisfied and there are no unused imports.
//
// This is declared as a variable rather than a function so goimports can
// easily be extended by adding a file with an init function.
var fixImports = fixImportsDefault

func fixImportsDefault(fset *token.FileSet, f *ast.File, filename string, env *ProcessEnv) error {
	fixes, err := getFixes(fset, f, filename, env)
	if err != nil {
		return err
	}
	apply(fset, f, fixes)
	return err
}

// getFixes gets the import fixes that need to be made to f in order to fix the imports.
// It does not modify the ast.
func getFixes(fset *token.FileSet, f *ast.File, filename string, env *ProcessEnv) ([]*ImportFix, error) {
	abs, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	srcDir := filepath.Dir(abs)
	if env.Logf != nil {
		env.Logf("fixImports(filename=%q), abs=%q, srcDir=%q ...", filename, abs, srcDir)
	}

	// First pass: looking only at f, and using the naive algorithm to
	// derive package names from import paths, see if the file is already
	// complete. We can't add any imports yet, because we don't know
	// if missing references are actually package vars.
	p := &pass{fset: fset, f: f, srcDir: srcDir, env: env}
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	otherFiles := parseOtherFiles(fset, srcDir, filename)

	// Second pass: add information from other files in the same package,
	// like their package vars and imports.
	p.otherFiles = otherFiles
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	// Now we can try adding imports from the stdlib.
	p.assumeSiblingImportsValid()
	addStdlibCandidates(p, p.missingRefs)
	if fixes, done := p.fix(); done {
		return fixes, nil
	}

	// Third pass: get real package names where we had previously used
	// the naive algorithm.
	p = &pass{fset: fset, f: f, srcDir: srcDir, env: env}
	p.loadRealPackageNames = true
	p.otherFiles = otherFiles
	if fixes, done := p.load(); done {
		return fixes, nil
	}

	if err := addStdlibCandidates(p, p.missingRefs); err != nil {
		return nil, err
	}
	p.assumeSiblingImportsValid()
	if fixes, done := p.fix(); done {
		return fixes, nil
	}

	// Go look for candidates in $GOPATH, etc. We don't necessarily load
	// the real exports of sibling imports, so keep assuming their contents.
	if err := addExternalCandidates(p, p.missingRefs, filename); err != nil {
		return nil, err
	}

	p.lastTry = true
	fixes, _ := p.fix()
	return fixes, nil
}

// MaxRelevance is the highest relevance, used for the standard library.
// Chosen arbitrarily to match pre-existing gopls code.
const MaxRelevance = 7.0

// getCandidatePkgs works with the passed callback to find all acceptable packages.
// It deduplicates by import path, and uses a cached stdlib rather than reading
// from disk.
func getCandidatePkgs(ctx context.Context, wrappedCallback *scanCallback, filename, filePkg string, env *ProcessEnv) error {
	notSelf := func(p *pkg) bool {
		return p.packageName != filePkg || p.dir != filepath.Dir(filename)
	}
	goenv, err := env.goEnv()
	if err != nil {
		return err
	}

	var mu sync.Mutex // to guard asynchronous access to dupCheck
	dupCheck := map[string]struct{}{}

	// Start off with the standard library.
	for importPath, exports := range stdlib {
		p := &pkg{
			dir:             filepath.Join(goenv["GOROOT"], "src", importPath),
			importPathShort: importPath,
			packageName:     path.Base(importPath),
			relevance:       MaxRelevance,
		}
		dupCheck[importPath] = struct{}{}
		if notSelf(p) && wrappedCallback.dirFound(p) && wrappedCallback.packageNameLoaded(p) {
			wrappedCallback.exportsLoaded(p, exports)
		}
	}

	scanFilter := &scanCallback{
		rootFound: func(root gopathwalk.Root) bool {
			// Exclude goroot results -- getting them is relatively expensive, not cached,
			// and generally redundant with the in-memory version.
			return root.Type != gopathwalk.RootGOROOT && wrappedCallback.rootFound(root)
		},
		dirFound: wrappedCallback.dirFound,
		packageNameLoaded: func(pkg *pkg) bool {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := dupCheck[pkg.importPathShort]; ok {
				return false
			}
			dupCheck[pkg.importPathShort] = struct{}{}
			return notSelf(pkg) && wrappedCallback.packageNameLoaded(pkg)
		},
		exportsLoaded: func(pkg *pkg, exports []string) {
			// If we're an x_test, load the package under test's test variant.
			if strings.HasSuffix(filePkg, "_test") && pkg.dir == filepath.Dir(filename) {
				var err error
				_, exports, err = loadExportsFromFiles(ctx, env, pkg.dir, true)
				if err != nil {
					return
				}
			}
			wrappedCallback.exportsLoaded(pkg, exports)
		},
	}
	resolver, err := env.GetResolver()
	if err != nil {
		return err
	}
	return resolver.scan(ctx, scanFilter)
}

func ScoreImportPaths(ctx context.Context, env *ProcessEnv, paths []string) (map[string]float64, error) {
	result := make(map[string]float64)
	resolver, err := env.GetResolver()
	if err != nil {
		return nil, err
	}
	for _, path := range paths {
		result[path] = resolver.scoreImportPath(ctx, path)
	}
	return result, nil
}

func PrimeCache(ctx context.Context, env *ProcessEnv) error {
	// Fully scan the disk for directories, but don't actually read any Go files.
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			return false
		},
		packageNameLoaded: func(pkg *pkg) bool {
			return false
		},
	}
	return getCandidatePkgs(ctx, callback, "", "", env)
}

func candidateImportName(pkg *pkg) string {
	if ImportPathToAssumedName(pkg.importPathShort) != pkg.packageName {
		return pkg.packageName
	}
	return ""
}

// GetAllCandidates calls wrapped for each package whose name starts with
// searchPrefix, and can be imported from filename with the package name filePkg.
func GetAllCandidates(ctx context.Context, wrapped func(ImportFix), searchPrefix, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			if !canUse(filename, pkg.dir) {
				return false
			}
			// Try the assumed package name first, then a simpler path match
			// in case of packages named vN, which are not uncommon.
			return strings.HasPrefix(ImportPathToAssumedName(pkg.importPathShort), searchPrefix) ||
				strings.HasPrefix(path.Base(pkg.importPathShort), searchPrefix)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			if !strings.HasPrefix(pkg.packageName, searchPrefix) {
				return false
			}
			wrapped(ImportFix{
				StmtInfo: ImportInfo{
					ImportPath: pkg.importPathShort,
					Name:       candidateImportName(pkg),
				},
				IdentName: pkg.packageName,
				FixType:   AddImport,
				Relevance: pkg.relevance,
			})
			return false
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

// GetImportPaths calls wrapped for each package whose import path starts with
// searchPrefix, and can be imported from filename with the package name filePkg.
func GetImportPaths(ctx context.Context, wrapped func(ImportFix), searchPrefix, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			if !canUse(filename, pkg.dir) {
				return false
			}
			return strings.HasPrefix(pkg.importPathShort, searchPrefix)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			wrapped(ImportFix{
				StmtInfo: ImportInfo{
					ImportPath: pkg.importPathShort,
					Name:       candidateImportName(pkg),
				},
				IdentName: pkg.packageName,
				FixType:   AddImport,
				Relevance: pkg.relevance,
			})
			return false
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

// A PackageExport is a package and its exports.
type PackageExport struct {
	Fix     *ImportFix
	Exports []string
}

// GetPackageExports returns all known packages with name pkg and their exports.
func GetPackageExports(ctx context.Context, wrapped func(PackageExport), searchPkg, filename, filePkg string, env *ProcessEnv) error {
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true
		},
		dirFound: func(pkg *pkg) bool {
			return pkgIsCandidate(filename, references{searchPkg: nil}, pkg)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			return pkg.packageName == searchPkg
		},
		exportsLoaded: func(pkg *pkg, exports []string) {
			sort.Strings(exports)
			wrapped(PackageExport{
				Fix: &ImportFix{
					StmtInfo: ImportInfo{
						ImportPath: pkg.importPathShort,
						Name:       candidateImportName(pkg),
					},
					IdentName: pkg.packageName,
					FixType:   AddImport,
					Relevance: pkg.relevance,
				},
				Exports: exports,
			})
		},
	}
	return getCandidatePkgs(ctx, callback, filename, filePkg, env)
}

var RequiredGoEnvVars = []string{"GO111MODULE", "GOFLAGS", "GOINSECURE", "GOMOD", "GOMODCACHE", "GONOPROXY", "GONOSUMDB", "GOPATH", "GOPROXY", "GOROOT", "GOSUMDB", "GOWORK"}

// ProcessEnv contains environment variables and settings that affect the use of
// the go command, the go/build package, etc.
type ProcessEnv struct {
	GocmdRunner *gocommand.Runner

	BuildFlags []string
	ModFlag    string
	ModFile    string

	// Env overrides the OS environment, and can be used to specify
	// GOPROXY, GO111MODULE, etc. PATH cannot be set here, because
	// exec.Command will not honor it.
	// Specifying all of RequiredGoEnvVars avoids a call to `go env`.
	Env map[string]string

	WorkingDir string

	// If Logf is non-nil, debug logging is enabled through this function.
	Logf func(format string, args ...interface{})

	initialized bool

	resolver Resolver
}

func (e *ProcessEnv) goEnv() (map[string]string, error) {
	if err := e.init(); err != nil {
		return nil, err
	}
	return e.Env, nil
}

func (e *ProcessEnv) matchFile(dir, name string) (bool, error) {
	bctx, err := e.buildContext()
	if err != nil {
		return false, err
	}
	return bctx.MatchFile(dir, name)
}

// CopyConfig copies the env's configuration into a new env.
func (e *ProcessEnv) CopyConfig() *ProcessEnv {
	copy := &ProcessEnv{
		GocmdRunner: e.GocmdRunner,
		initialized: e.initialized,
		BuildFlags:  e.BuildFlags,
		Logf:        e.Logf,
		WorkingDir:  e.WorkingDir,
		resolver:    nil,
		Env:         map[string]string{},
	}
	for k, v := range e.Env {
		copy.Env[k] = v
	}
	return copy
}

func (e *ProcessEnv) init() error {
	if e.initialized {
		return nil
	}

	foundAllRequired := true
	for _, k := range RequiredGoEnvVars {
		if _, ok := e.Env[k]; !ok {
			foundAllRequired = false
			break
		}
	}
	if foundAllRequired {
		e.initialized = true
		return nil
	}

	if e.Env == nil {
		e.Env = map[string]string{}
	}

	goEnv := map[string]string{}
	stdout, err := e.invokeGo(context.TODO(), "env", append([]string{"-json"}, RequiredGoEnvVars...)...)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(stdout.Bytes(), &goEnv); err != nil {
		return err
	}
	for k, v := range goEnv {
		e.Env[k] = v
	}
	e.initialized = true
	return nil
}

func (e *ProcessEnv) env() []string {
	var env []string // the gocommand package will prepend os.Environ.
	for k, v := range e.Env {
		env = append(env, k+"="+v)
	}
	return env
}

func (e *ProcessEnv) GetResolver() (Resolver, error) {
	if e.resolver != nil {
		return e.resolver, nil
	}
	if err := e.init(); err != nil {
		return nil, err
	}
	if len(e.Env["GOMOD"]) == 0 && len(e.Env["GOWORK"]) == 0 {
		e.resolver = newGopathResolver(e)
		return e.resolver, nil
	}
	e.resolver = newModuleResolver(e)
	return e.resolver, nil
}

func (e *ProcessEnv) buildContext() (*build.Context, error) {
	ctx := build.Default
	goenv, err := e.goEnv()
	if err != nil {
		return nil, err
	}
	ctx.GOROOT = goenv["GOROOT"]
	ctx.GOPATH = goenv["GOPATH"]

	// As of Go 1.14, build.Context has a Dir field
	// (see golang.org/issue/34860).
	// Populate it only if present.
	rc := reflect.ValueOf(&ctx).Elem()
	dir := rc.FieldByName("Dir")
	if dir.IsValid() && dir.Kind() == reflect.String {
		dir.SetString(e.WorkingDir)
	}

	// Since Go 1.11, go/build.Context.Import may invoke 'go list' depending on
	// the value in GO111MODULE in the process's environment. We always want to
	// run in GOPATH mode when calling Import, so we need to prevent this from
	// happening. In Go 1.16, GO111MODULE defaults to "on", so this problem comes
	// up more frequently.
	//
	// HACK: setting any of the Context I/O hooks prevents Import from invoking
	// 'go list', regardless of GO111MODULE. This is undocumented, but it's
	// unlikely to change before GOPATH support is removed.
	ctx.ReadDir = ioutil.ReadDir

	return &ctx, nil
}

func (e *ProcessEnv) invokeGo(ctx context.Context, verb string, args ...string) (*bytes.Buffer, error) {
	inv := gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		BuildFlags: e.BuildFlags,
		Env:        e.env(),
		Logf:       e.Logf,
		WorkingDir: e.WorkingDir,
	}
	return e.GocmdRunner.Run(ctx, inv)
}

func addStdlibCandidates(pass *pass, refs references) error {
	goenv, err := pass.env.goEnv()
	if err != nil {
		return err
	}
	add := func(pkg string) {
		// Prevent self-imports.
		if path.Base(pkg) == pass.f.Name.Name && filepath.Join(goenv["GOROOT"], "src", pkg) == pass.srcDir {
			return
		}
		exports := copyExports(stdlib[pkg])
		pass.addCandidate(
			&ImportInfo{ImportPath: pkg},
			&packageInfo{name: path.Base(pkg), exports: exports})
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
	return nil
}

// A Resolver does the build-system-specific parts of goimports.
type Resolver interface {
	// loadPackageNames loads the package names in importPaths.
	loadPackageNames(importPaths []string, srcDir string) (map[string]string, error)
	// scan works with callback to search for packages. See scanCallback for details.
	scan(ctx context.Context, callback *scanCallback) error
	// loadExports returns the set of exported symbols in the package at dir.
	// loadExports may be called concurrently.
	loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []string, error)
	// scoreImportPath returns the relevance for an import path.
	scoreImportPath(ctx context.Context, path string) float64

	ClearForNewScan()
}

// A scanCallback controls a call to scan and receives its results.
// In general, minor errors will be silently discarded; a user should not
// expect to receive a full series of calls for everything.
type scanCallback struct {
	// rootFound is called before scanning a new root dir. If it returns true,
	// the root will be scanned. Returning false will not necessarily prevent
	// directories from that root making it to dirFound.
	rootFound func(gopathwalk.Root) bool
	// dirFound is called when a directory is found that is possibly a Go package.
	// pkg will be populated with everything except packageName.
	// If it returns true, the package's name will be loaded.
	dirFound func(pkg *pkg) bool
	// packageNameLoaded is called when a package is found and its name is loaded.
	// If it returns true, the package's exports will be loaded.
	packageNameLoaded func(pkg *pkg) bool
	// exportsLoaded is called when a package's exports have been loaded.
	exportsLoaded func(pkg *pkg, exports []string)
}

func addExternalCandidates(pass *pass, refs references, filename string) error {
	var mu sync.Mutex
	found := make(map[string][]pkgDistance)
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true // We want everything.
		},
		dirFound: func(pkg *pkg) bool {
			return pkgIsCandidate(filename, refs, pkg)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			if _, want := refs[pkg.packageName]; !want {
				return false
			}
			if pkg.dir == pass.srcDir && pass.f.Name.Name == pkg.packageName {
				// The candidate is in the same directory and has the
				// same package name. Don't try to import ourselves.
				return false
			}
			if !canUse(filename, pkg.dir) {
				return false
			}
			mu.Lock()
			defer mu.Unlock()
			found[pkg.packageName] = append(found[pkg.packageName], pkgDistance{pkg, distance(pass.srcDir, pkg.dir)})
			return false // We'll do our own loading after we sort.
		},
	}
	resolver, err := pass.env.GetResolver()
	if err != nil {
		return err
	}
	if err = resolver.scan(context.Background(), callback); err != nil {
		return err
	}

	// Search for imports matching potential package references.
	type result struct {
		imp *ImportInfo
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

			found, err := findImport(ctx, pass, found[pkgName], pkgName, symbols, filename)

			if err != nil {
				firstErrOnce.Do(func() {
					firstErr = err
					cancel()
				})
				return
			}

			if found == nil {
				return // No matching package.
			}

			imp := &ImportInfo{
				ImportPath: found.importPathShort,
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

// notIdentifier reports whether ch is an invalid identifier character.
func notIdentifier(ch rune) bool {
	return !('a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' ||
		'0' <= ch && ch <= '9' ||
		ch == '_' ||
		ch >= utf8.RuneSelf && (unicode.IsLetter(ch) || unicode.IsDigit(ch)))
}

// ImportPathToAssumedName returns the assumed package name of an import path.
// It does this using only string parsing of the import path.
// It picks the last element of the path that does not look like a major
// version, and then picks the valid identifier off the start of that element.
// It is used to determine if a local rename should be added to an import for
// clarity.
// This function could be moved to a standard package and exported if we want
// for use in other tools.
func ImportPathToAssumedName(importPath string) string {
	base := path.Base(importPath)
	if strings.HasPrefix(base, "v") {
		if _, err := strconv.Atoi(base[1:]); err == nil {
			dir := path.Dir(importPath)
			if dir != "." {
				base = path.Base(dir)
			}
		}
	}
	base = strings.TrimPrefix(base, "go-")
	if i := strings.IndexFunc(base, notIdentifier); i >= 0 {
		base = base[:i]
	}
	return base
}

// gopathResolver implements resolver for GOPATH workspaces.
type gopathResolver struct {
	env      *ProcessEnv
	walked   bool
	cache    *dirInfoCache
	scanSema chan struct{} // scanSema prevents concurrent scans.
}

func newGopathResolver(env *ProcessEnv) *gopathResolver {
	r := &gopathResolver{
		env: env,
		cache: &dirInfoCache{
			dirs:      map[string]*directoryPackageInfo{},
			listeners: map[*int]cacheListener{},
		},
		scanSema: make(chan struct{}, 1),
	}
	r.scanSema <- struct{}{}
	return r
}

func (r *gopathResolver) ClearForNewScan() {
	<-r.scanSema
	r.cache = &dirInfoCache{
		dirs:      map[string]*directoryPackageInfo{},
		listeners: map[*int]cacheListener{},
	}
	r.walked = false
	r.scanSema <- struct{}{}
}

func (r *gopathResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	names := map[string]string{}
	bctx, err := r.env.buildContext()
	if err != nil {
		return nil, err
	}
	for _, path := range importPaths {
		names[path] = importPathToName(bctx, path, srcDir)
	}
	return names, nil
}

// importPathToName finds out the actual package name, as declared in its .go files.
func importPathToName(bctx *build.Context, importPath, srcDir string) string {
	// Fast path for standard library without going to disk.
	if _, ok := stdlib[importPath]; ok {
		return path.Base(importPath) // stdlib packages always match their paths.
	}

	buildPkg, err := bctx.Import(importPath, srcDir, build.FindOnly)
	if err != nil {
		return ""
	}
	pkgName, err := packageDirToName(buildPkg.Dir)
	if err != nil {
		return ""
	}
	return pkgName
}

// packageDirToName is a faster version of build.Import if
// the only thing desired is the package name. Given a directory,
// packageDirToName then only parses one file in the package,
// trusting that the files in the directory are consistent.
func packageDirToName(dir string) (packageName string, err error) {
	d, err := os.Open(dir)
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
		fullFile := filepath.Join(dir, name)

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

type pkg struct {
	dir             string  // absolute file path to pkg directory ("/usr/lib/go/src/net/http")
	importPathShort string  // vendorless import path ("net/http", "a/b")
	packageName     string  // package name loaded from source if requested
	relevance       float64 // a weakly-defined score of how relevant a package is. 0 is most relevant.
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

func (r *gopathResolver) scan(ctx context.Context, callback *scanCallback) error {
	add := func(root gopathwalk.Root, dir string) {
		// We assume cached directories have not changed. We can skip them and their
		// children.
		if _, ok := r.cache.Load(dir); ok {
			return
		}

		importpath := filepath.ToSlash(dir[len(root.Path)+len("/"):])
		info := directoryPackageInfo{
			status:                 directoryScanned,
			dir:                    dir,
			rootType:               root.Type,
			nonCanonicalImportPath: VendorlessPath(importpath),
		}
		r.cache.Store(dir, info)
	}
	processDir := func(info directoryPackageInfo) {
		// Skip this directory if we were not able to get the package information successfully.
		if scanned, err := info.reachedStatus(directoryScanned); !scanned || err != nil {
			return
		}

		p := &pkg{
			importPathShort: info.nonCanonicalImportPath,
			dir:             info.dir,
			relevance:       MaxRelevance - 1,
		}
		if info.rootType == gopathwalk.RootGOROOT {
			p.relevance = MaxRelevance
		}

		if !callback.dirFound(p) {
			return
		}
		var err error
		p.packageName, err = r.cache.CachePackageName(info)
		if err != nil {
			return
		}

		if !callback.packageNameLoaded(p) {
			return
		}
		if _, exports, err := r.loadExports(ctx, p, false); err == nil {
			callback.exportsLoaded(p, exports)
		}
	}
	stop := r.cache.ScanAndListen(ctx, processDir)
	defer stop()

	goenv, err := r.env.goEnv()
	if err != nil {
		return err
	}
	var roots []gopathwalk.Root
	roots = append(roots, gopathwalk.Root{filepath.Join(goenv["GOROOT"], "src"), gopathwalk.RootGOROOT})
	for _, p := range filepath.SplitList(goenv["GOPATH"]) {
		roots = append(roots, gopathwalk.Root{filepath.Join(p, "src"), gopathwalk.RootGOPATH})
	}
	// The callback is not necessarily safe to use in the goroutine below. Process roots eagerly.
	roots = filterRoots(roots, callback.rootFound)
	// We can't cancel walks, because we need them to finish to have a usable
	// cache. Instead, run them in a separate goroutine and detach.
	scanDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			return
		case <-r.scanSema:
		}
		defer func() { r.scanSema <- struct{}{} }()
		gopathwalk.Walk(roots, add, gopathwalk.Options{Logf: r.env.Logf, ModulesEnabled: false})
		close(scanDone)
	}()
	select {
	case <-ctx.Done():
	case <-scanDone:
	}
	return nil
}

func (r *gopathResolver) scoreImportPath(ctx context.Context, path string) float64 {
	if _, ok := stdlib[path]; ok {
		return MaxRelevance
	}
	return MaxRelevance - 1
}

func filterRoots(roots []gopathwalk.Root, include func(gopathwalk.Root) bool) []gopathwalk.Root {
	var result []gopathwalk.Root
	for _, root := range roots {
		if !include(root) {
			continue
		}
		result = append(result, root)
	}
	return result
}

func (r *gopathResolver) loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []string, error) {
	if info, ok := r.cache.Load(pkg.dir); ok && !includeTest {
		return r.cache.CacheExports(ctx, r.env, info)
	}
	return loadExportsFromFiles(ctx, r.env, pkg.dir, includeTest)
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

func loadExportsFromFiles(ctx context.Context, env *ProcessEnv, dir string, includeTest bool) (string, []string, error) {
	// Look for non-test, buildable .go files which could provide exports.
	all, err := ioutil.ReadDir(dir)
	if err != nil {
		return "", nil, err
	}
	var files []os.FileInfo
	for _, fi := range all {
		name := fi.Name()
		if !strings.HasSuffix(name, ".go") || (!includeTest && strings.HasSuffix(name, "_test.go")) {
			continue
		}
		match, err := env.matchFile(dir, fi.Name())
		if err != nil || !match {
			continue
		}
		files = append(files, fi)
	}

	if len(files) == 0 {
		return "", nil, fmt.Errorf("dir %v contains no buildable, non-test .go files", dir)
	}

	var pkgName string
	var exports []string
	fset := token.NewFileSet()
	for _, fi := range files {
		select {
		case <-ctx.Done():
			return "", nil, ctx.Err()
		default:
		}

		fullFile := filepath.Join(dir, fi.Name())
		f, err := parser.ParseFile(fset, fullFile, nil, 0)
		if err != nil {
			if env.Logf != nil {
				env.Logf("error parsing %v: %v", fullFile, err)
			}
			continue
		}
		if f.Name.Name == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by MatchFile above.
			continue
		}
		if includeTest && strings.HasSuffix(f.Name.Name, "_test") {
			// x_test package. We want internal test files only.
			continue
		}
		pkgName = f.Name.Name
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				exports = append(exports, name)
			}
		}
	}

	if env.Logf != nil {
		sortedExports := append([]string(nil), exports...)
		sort.Strings(sortedExports)
		env.Logf("loaded exports in dir %v (package %v): %v", dir, pkgName, strings.Join(sortedExports, ", "))
	}
	return pkgName, exports, nil
}

// findImport searches for a package with the given symbols.
// If no package is found, findImport returns ("", false, nil)
func findImport(ctx context.Context, pass *pass, candidates []pkgDistance, pkgName string, symbols map[string]bool, filename string) (*pkg, error) {
	// Sort the candidates by their import package length,
	// assuming that shorter package names are better than long
	// ones.  Note that this sorts by the de-vendored name, so
	// there's no "penalty" for vendoring.
	sort.Sort(byDistanceOrImportPathShortLength(candidates))
	if pass.env.Logf != nil {
		for i, c := range candidates {
			pass.env.Logf("%s candidate %d/%d: %v in %v", pkgName, i+1, len(candidates), c.pkg.importPathShort, c.pkg.dir)
		}
	}
	resolver, err := pass.env.GetResolver()
	if err != nil {
		return nil, err
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

				if pass.env.Logf != nil {
					pass.env.Logf("loading exports in dir %s (seeking package %s)", c.pkg.dir, pkgName)
				}
				// If we're an x_test, load the package under test's test variant.
				includeTest := strings.HasSuffix(pass.f.Name.Name, "_test") && c.pkg.dir == pass.srcDir
				_, exports, err := resolver.loadExports(ctx, c.pkg, includeTest)
				if err != nil {
					if pass.env.Logf != nil {
						pass.env.Logf("loading exports in dir %s (seeking package %s): %v", c.pkg.dir, pkgName, err)
					}
					resc <- nil
					return
				}

				exportsMap := make(map[string]bool, len(exports))
				for _, sym := range exports {
					exportsMap[sym] = true
				}

				// If it doesn't have the right
				// symbols, send nil to mean no match.
				for symbol := range symbols {
					if !exportsMap[symbol] {
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
		return pkg, nil
	}
	return nil, nil
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
func pkgIsCandidate(filename string, refs references, pkg *pkg) bool {
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
	for pkgIdent := range refs {
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

func copyExports(pkg []string) map[string]bool {
	m := make(map[string]bool, len(pkg))
	for _, v := range pkg {
		m[v] = true
	}
	return m
}
