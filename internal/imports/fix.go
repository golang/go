// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/gopathwalk"
)

// importToGroup is a list of functions which map from an import path to
// a group number.
var importToGroup = []func(env *ProcessEnv, importPath string) (num int, ok bool){
	func(env *ProcessEnv, importPath string) (num int, ok bool) {
		if env.LocalPrefix == "" {
			return
		}
		for _, p := range strings.Split(env.LocalPrefix, ",") {
			if strings.HasPrefix(importPath, p) || strings.TrimSuffix(p, "/") == importPath {
				return 3, true
			}
		}
		return
	},
	func(_ *ProcessEnv, importPath string) (num int, ok bool) {
		if strings.HasPrefix(importPath, "appengine") {
			return 2, true
		}
		return
	},
	func(_ *ProcessEnv, importPath string) (num int, ok bool) {
		if strings.Contains(importPath, ".") {
			return 1, true
		}
		return
	},
}

func importGroup(env *ProcessEnv, importPath string) int {
	for _, fn := range importToGroup {
		if n, ok := fn(env, importPath); ok {
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
	existingImports map[string]*importInfo
	allRefs         references
	missingRefs     references

	// Inputs to fix. These can be augmented between successive fix calls.
	lastTry       bool                    // indicates that this is the last call and fix should clean up as best it can.
	candidates    []*importInfo           // candidate imports in priority order.
	knownPackages map[string]*packageInfo // information about all known packages.
}

// loadPackageNames saves the package names for everything referenced by imports.
func (p *pass) loadPackageNames(imports []*importInfo) error {
	var unknown []string
	for _, imp := range imports {
		if _, ok := p.knownPackages[imp.importPath]; ok {
			continue
		}
		unknown = append(unknown, imp.importPath)
	}

	names, err := p.env.getResolver().loadPackageNames(unknown, p.srcDir)
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
func (p *pass) importIdentifier(imp *importInfo) string {
	if imp.name != "" {
		return imp.name
	}
	known := p.knownPackages[imp.importPath]
	if known != nil && known.name != "" {
		return known.name
	}
	return importPathToAssumedName(imp.importPath)
}

// load reads in everything necessary to run a pass, and reports whether the
// file already has all the imports it needs. It fills in p.missingRefs with the
// file's missing symbols, if any, or removes unused imports if not.
func (p *pass) load() bool {
	p.knownPackages = map[string]*packageInfo{}
	p.missingRefs = references{}
	p.existingImports = map[string]*importInfo{}

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
			if p.env.Debug {
				log.Printf("loading package names: %v", err)
			}
			return false
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
	for left, rights := range p.missingRefs {
		if imp := p.findMissingImport(left, rights); imp != nil {
			selected = append(selected, imp)
		}
	}

	if !p.lastTry && len(selected) != len(p.missingRefs) {
		return false
	}

	// Found everything, or giving up. Add the new imports and remove any unused.
	for _, imp := range p.existingImports {
		// We deliberately ignore globals here, because we can't be sure
		// they're in the same package. People do things like put multiple
		// main packages in the same directory, and we don't want to
		// remove imports if they happen to have the same name as a var in
		// a different package.
		if _, ok := p.allRefs[p.importIdentifier(imp)]; !ok {
			astutil.DeleteNamedImport(p.fset, p.f, imp.name, imp.importPath)
		}
	}

	for _, imp := range selected {
		astutil.AddNamedImport(p.fset, p.f, imp.name, imp.importPath)
	}

	if p.loadRealPackageNames {
		for _, imp := range p.f.Imports {
			if imp.Name != nil {
				continue
			}
			path := strings.Trim(imp.Path.Value, `""`)
			ident := p.importIdentifier(&importInfo{importPath: path})
			if ident != importPathToAssumedName(path) {
				imp.Name = &ast.Ident{Name: ident, NamePos: imp.Pos()}
			}
		}
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
					rights = stdlib[imp.importPath]
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
		if existing.name == "" {
			existing.name = pkg.name
		}
		for export := range pkg.exports {
			existing.exports[export] = true
		}
	} else {
		p.knownPackages[imp.importPath] = pkg
	}
}

// fixImports adds and removes imports from f so that all its references are
// satisfied and there are no unused imports.
//
// This is declared as a variable rather than a function so goimports can
// easily be extended by adding a file with an init function.
var fixImports = fixImportsDefault

func fixImportsDefault(fset *token.FileSet, f *ast.File, filename string, env *ProcessEnv) error {
	abs, err := filepath.Abs(filename)
	if err != nil {
		return err
	}
	srcDir := filepath.Dir(abs)
	if env.Debug {
		log.Printf("fixImports(filename=%q), abs=%q, srcDir=%q ...", filename, abs, srcDir)
	}

	// First pass: looking only at f, and using the naive algorithm to
	// derive package names from import paths, see if the file is already
	// complete. We can't add any imports yet, because we don't know
	// if missing references are actually package vars.
	p := &pass{fset: fset, f: f, srcDir: srcDir}
	if p.load() {
		return nil
	}

	otherFiles := parseOtherFiles(fset, srcDir, filename)

	// Second pass: add information from other files in the same package,
	// like their package vars and imports.
	p.otherFiles = otherFiles
	if p.load() {
		return nil
	}

	// Now we can try adding imports from the stdlib.
	p.assumeSiblingImportsValid()
	addStdlibCandidates(p, p.missingRefs)
	if p.fix() {
		return nil
	}

	// Third pass: get real package names where we had previously used
	// the naive algorithm. This is the first step that will use the
	// environment, so we provide it here for the first time.
	p = &pass{fset: fset, f: f, srcDir: srcDir, env: env}
	p.loadRealPackageNames = true
	p.otherFiles = otherFiles
	if p.load() {
		return nil
	}

	addStdlibCandidates(p, p.missingRefs)
	p.assumeSiblingImportsValid()
	if p.fix() {
		return nil
	}

	// Go look for candidates in $GOPATH, etc. We don't necessarily load
	// the real exports of sibling imports, so keep assuming their contents.
	if err := addExternalCandidates(p, p.missingRefs, filename); err != nil {
		return err
	}

	p.lastTry = true
	p.fix()
	return nil
}

// ProcessEnv contains environment variables and settings that affect the use of
// the go command, the go/build package, etc.
type ProcessEnv struct {
	LocalPrefix string
	Debug       bool

	// If non-empty, these will be used instead of the
	// process-wide values.
	GOPATH, GOROOT, GO111MODULE, GOPROXY, GOFLAGS string
	WorkingDir                                    string

	// If true, use go/packages regardless of the environment.
	ForceGoPackages bool

	resolver resolver
}

func (e *ProcessEnv) env() []string {
	env := os.Environ()
	add := func(k, v string) {
		if v != "" {
			env = append(env, k+"="+v)
		}
	}
	add("GOPATH", e.GOPATH)
	add("GOROOT", e.GOROOT)
	add("GO111MODULE", e.GO111MODULE)
	add("GOPROXY", e.GOPROXY)
	add("GOFLAGS", e.GOFLAGS)
	if e.WorkingDir != "" {
		add("PWD", e.WorkingDir)
	}
	return env
}

func (e *ProcessEnv) getResolver() resolver {
	if e.resolver != nil {
		return e.resolver
	}
	if e.ForceGoPackages {
		return &goPackagesResolver{env: e}
	}

	out, err := e.invokeGo("env", "GOMOD")
	if err != nil || len(bytes.TrimSpace(out.Bytes())) == 0 {
		return &gopathResolver{env: e}
	}
	return &moduleResolver{env: e}
}

func (e *ProcessEnv) newPackagesConfig(mode packages.LoadMode) *packages.Config {
	return &packages.Config{
		Mode: mode,
		Dir:  e.WorkingDir,
		Env:  e.env(),
	}
}

func (e *ProcessEnv) buildContext() *build.Context {
	ctx := build.Default
	ctx.GOROOT = e.GOROOT
	ctx.GOPATH = e.GOPATH
	return &ctx
}

func (e *ProcessEnv) invokeGo(args ...string) (*bytes.Buffer, error) {
	cmd := exec.Command("go", args...)
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Env = e.env()
	cmd.Dir = e.WorkingDir

	if e.Debug {
		defer func(start time.Time) { log.Printf("%s for %v", time.Since(start), cmdDebugStr(cmd)) }(time.Now())
	}
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("running go: %v (stderr:\n%s)", err, stderr)
	}
	return stdout, nil
}

func cmdDebugStr(cmd *exec.Cmd) string {
	env := make(map[string]string)
	for _, kv := range cmd.Env {
		split := strings.Split(kv, "=")
		k, v := split[0], split[1]
		env[k] = v
	}

	return fmt.Sprintf("GOROOT=%v GOPATH=%v GO111MODULE=%v GOPROXY=%v PWD=%v go %v", env["GOROOT"], env["GOPATH"], env["GO111MODULE"], env["GOPROXY"], env["PWD"], cmd.Args)
}

func addStdlibCandidates(pass *pass, refs references) {
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

// A resolver does the build-system-specific parts of goimports.
type resolver interface {
	// loadPackageNames loads the package names in importPaths.
	loadPackageNames(importPaths []string, srcDir string) (map[string]string, error)
	// scan finds (at least) the packages satisfying refs. The returned slice is unordered.
	scan(refs references) ([]*pkg, error)
}

// gopathResolver implements resolver for GOPATH and module workspaces using go/packages.
type goPackagesResolver struct {
	env *ProcessEnv
}

func (r *goPackagesResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	cfg := r.env.newPackagesConfig(packages.LoadFiles)
	pkgs, err := packages.Load(cfg, importPaths...)
	if err != nil {
		return nil, err
	}
	names := map[string]string{}
	for _, pkg := range pkgs {
		names[VendorlessPath(pkg.PkgPath)] = pkg.Name
	}
	// We may not have found all the packages. Guess the rest.
	for _, path := range importPaths {
		if _, ok := names[path]; ok {
			continue
		}
		names[path] = importPathToAssumedName(path)
	}
	return names, nil

}

func (r *goPackagesResolver) scan(refs references) ([]*pkg, error) {
	var loadQueries []string
	for pkgName := range refs {
		loadQueries = append(loadQueries, "iamashamedtousethedisabledqueryname="+pkgName)
	}
	sort.Strings(loadQueries)
	cfg := r.env.newPackagesConfig(packages.LoadFiles)
	goPackages, err := packages.Load(cfg, loadQueries...)
	if err != nil {
		return nil, err
	}

	var scan []*pkg
	for _, goPackage := range goPackages {
		scan = append(scan, &pkg{
			dir:             filepath.Dir(goPackage.CompiledGoFiles[0]),
			importPathShort: VendorlessPath(goPackage.PkgPath),
			goPackage:       goPackage,
		})
	}
	return scan, nil
}

func addExternalCandidates(pass *pass, refs references, filename string) error {
	dirScan, err := pass.env.getResolver().scan(refs)
	if err != nil {
		return err
	}

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

			found, err := findImport(ctx, pass.env, dirScan, pkgName, symbols, filename)

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

			imp := &importInfo{
				importPath: found.importPathShort,
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

// importPathToAssumedName returns the assumed package name of an import path.
// It does this using only string parsing of the import path.
// It picks the last element of the path that does not look like a major
// version, and then picks the valid identifier off the start of that element.
// It is used to determine if a local rename should be added to an import for
// clarity.
// This function could be moved to a standard package and exported if we want
// for use in other tools.
func importPathToAssumedName(importPath string) string {
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
	env *ProcessEnv
}

func (r *gopathResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	names := map[string]string{}
	for _, path := range importPaths {
		names[path] = importPathToName(r.env, path, srcDir)
	}
	return names, nil
}

// importPathToNameGoPath finds out the actual package name, as declared in its .go files.
// If there's a problem, it returns "".
func importPathToName(env *ProcessEnv, importPath, srcDir string) (packageName string) {
	// Fast path for standard library without going to disk.
	if _, ok := stdlib[importPath]; ok {
		return path.Base(importPath) // stdlib packages always match their paths.
	}

	buildPkg, err := env.buildContext().Import(importPath, srcDir, build.FindOnly)
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
// the only thing desired is the package name. It uses build.FindOnly
// to find the directory and then only parses one file in the package,
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
	goPackage       *packages.Package
	dir             string // absolute file path to pkg directory ("/usr/lib/go/src/net/http")
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

func (r *gopathResolver) scan(_ references) ([]*pkg, error) {
	dupCheck := make(map[string]bool)
	var result []*pkg

	var mu sync.Mutex

	add := func(root gopathwalk.Root, dir string) {
		mu.Lock()
		defer mu.Unlock()

		if _, dup := dupCheck[dir]; dup {
			return
		}
		dupCheck[dir] = true
		importpath := filepath.ToSlash(dir[len(root.Path)+len("/"):])
		result = append(result, &pkg{
			importPathShort: VendorlessPath(importpath),
			dir:             dir,
		})
	}
	gopathwalk.Walk(gopathwalk.SrcDirsRoots(r.env.buildContext()), add, gopathwalk.Options{Debug: r.env.Debug, ModulesEnabled: false})
	return result, nil
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
func loadExports(ctx context.Context, env *ProcessEnv, expectPackage string, pkg *pkg) (map[string]bool, error) {
	if env.Debug {
		log.Printf("loading exports in dir %s (seeking package %s)", pkg.dir, expectPackage)
	}
	if pkg.goPackage != nil {
		exports := map[string]bool{}
		fset := token.NewFileSet()
		for _, fname := range pkg.goPackage.CompiledGoFiles {
			f, err := parser.ParseFile(fset, fname, nil, 0)
			if err != nil {
				return nil, fmt.Errorf("parsing %s: %v", fname, err)
			}
			for name := range f.Scope.Objects {
				if ast.IsExported(name) {
					exports[name] = true
				}
			}
		}
		return exports, nil
	}

	exports := make(map[string]bool)

	// Look for non-test, buildable .go files which could provide exports.
	all, err := ioutil.ReadDir(pkg.dir)
	if err != nil {
		return nil, err
	}
	var files []os.FileInfo
	for _, fi := range all {
		name := fi.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		match, err := env.buildContext().MatchFile(pkg.dir, fi.Name())
		if err != nil || !match {
			continue
		}
		files = append(files, fi)
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("dir %v contains no buildable, non-test .go files", pkg.dir)
	}

	fset := token.NewFileSet()
	for _, fi := range files {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		fullFile := filepath.Join(pkg.dir, fi.Name())
		f, err := parser.ParseFile(fset, fullFile, nil, 0)
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %v", fullFile, err)
		}
		pkgName := f.Name.Name
		if pkgName == "documentation" {
			// Special case from go/build.ImportDir, not
			// handled by MatchFile above.
			continue
		}
		if pkgName != expectPackage {
			return nil, fmt.Errorf("scan of dir %v is not expected package %v (actually %v)", pkg.dir, expectPackage, pkgName)
		}
		for name := range f.Scope.Objects {
			if ast.IsExported(name) {
				exports[name] = true
			}
		}
	}

	if env.Debug {
		exportList := make([]string, 0, len(exports))
		for k := range exports {
			exportList = append(exportList, k)
		}
		sort.Strings(exportList)
		log.Printf("loaded exports in dir %v (package %v): %v", pkg.dir, expectPackage, strings.Join(exportList, ", "))
	}
	return exports, nil
}

// findImport searches for a package with the given symbols.
// If no package is found, findImport returns ("", false, nil)
func findImport(ctx context.Context, env *ProcessEnv, dirScan []*pkg, pkgName string, symbols map[string]bool, filename string) (*pkg, error) {
	pkgDir, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	pkgDir = filepath.Dir(pkgDir)

	// Find candidate packages, looking only at their directory names first.
	var candidates []pkgDistance
	for _, pkg := range dirScan {
		if pkg.dir != pkgDir && pkgIsCandidate(filename, pkgName, pkg) {
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
	if env.Debug {
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

				exports, err := loadExports(ctx, env, pkgName, c.pkg)
				if err != nil {
					if env.Debug {
						log.Printf("loading exports in dir %s (seeking package %s): %v", c.pkg.dir, pkgName, err)
					}
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
