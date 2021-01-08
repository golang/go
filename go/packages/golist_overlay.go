// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/internal/gocommand"
)

// processGolistOverlay provides rudimentary support for adding
// files that don't exist on disk to an overlay. The results can be
// sometimes incorrect.
// TODO(matloob): Handle unsupported cases, including the following:
// - determining the correct package to add given a new import path
func (state *golistState) processGolistOverlay(response *responseDeduper) (modifiedPkgs, needPkgs []string, err error) {
	havePkgs := make(map[string]string) // importPath -> non-test package ID
	needPkgsSet := make(map[string]bool)
	modifiedPkgsSet := make(map[string]bool)

	pkgOfDir := make(map[string][]*Package)
	for _, pkg := range response.dr.Packages {
		// This is an approximation of import path to id. This can be
		// wrong for tests, vendored packages, and a number of other cases.
		havePkgs[pkg.PkgPath] = pkg.ID
		dir, err := commonDir(pkg.GoFiles)
		if err != nil {
			return nil, nil, err
		}
		if dir != "" {
			pkgOfDir[dir] = append(pkgOfDir[dir], pkg)
		}
	}

	// If no new imports are added, it is safe to avoid loading any needPkgs.
	// Otherwise, it's hard to tell which package is actually being loaded
	// (due to vendoring) and whether any modified package will show up
	// in the transitive set of dependencies (because new imports are added,
	// potentially modifying the transitive set of dependencies).
	var overlayAddsImports bool

	// If both a package and its test package are created by the overlay, we
	// need the real package first. Process all non-test files before test
	// files, and make the whole process deterministic while we're at it.
	var overlayFiles []string
	for opath := range state.cfg.Overlay {
		overlayFiles = append(overlayFiles, opath)
	}
	sort.Slice(overlayFiles, func(i, j int) bool {
		iTest := strings.HasSuffix(overlayFiles[i], "_test.go")
		jTest := strings.HasSuffix(overlayFiles[j], "_test.go")
		if iTest != jTest {
			return !iTest // non-tests are before tests.
		}
		return overlayFiles[i] < overlayFiles[j]
	})
	for _, opath := range overlayFiles {
		contents := state.cfg.Overlay[opath]
		base := filepath.Base(opath)
		dir := filepath.Dir(opath)
		var pkg *Package           // if opath belongs to both a package and its test variant, this will be the test variant
		var testVariantOf *Package // if opath is a test file, this is the package it is testing
		var fileExists bool
		isTestFile := strings.HasSuffix(opath, "_test.go")
		pkgName, ok := extractPackageName(opath, contents)
		if !ok {
			// Don't bother adding a file that doesn't even have a parsable package statement
			// to the overlay.
			continue
		}
		// If all the overlay files belong to a different package, change the
		// package name to that package.
		maybeFixPackageName(pkgName, isTestFile, pkgOfDir[dir])
	nextPackage:
		for _, p := range response.dr.Packages {
			if pkgName != p.Name && p.ID != "command-line-arguments" {
				continue
			}
			for _, f := range p.GoFiles {
				if !sameFile(filepath.Dir(f), dir) {
					continue
				}
				// Make sure to capture information on the package's test variant, if needed.
				if isTestFile && !hasTestFiles(p) {
					// TODO(matloob): Are there packages other than the 'production' variant
					// of a package that this can match? This shouldn't match the test main package
					// because the file is generated in another directory.
					testVariantOf = p
					continue nextPackage
				} else if !isTestFile && hasTestFiles(p) {
					// We're examining a test variant, but the overlaid file is
					// a non-test file. Because the overlay implementation
					// (currently) only adds a file to one package, skip this
					// package, so that we can add the file to the production
					// variant of the package. (https://golang.org/issue/36857
					// tracks handling overlays on both the production and test
					// variant of a package).
					continue nextPackage
				}
				if pkg != nil && p != pkg && pkg.PkgPath == p.PkgPath {
					// We have already seen the production version of the
					// for which p is a test variant.
					if hasTestFiles(p) {
						testVariantOf = pkg
					}
				}
				pkg = p
				if filepath.Base(f) == base {
					fileExists = true
				}
			}
		}
		// The overlay could have included an entirely new package or an
		// ad-hoc package. An ad-hoc package is one that we have manually
		// constructed from inadequate `go list` results for a file= query.
		// It will have the ID command-line-arguments.
		if pkg == nil || pkg.ID == "command-line-arguments" {
			// Try to find the module or gopath dir the file is contained in.
			// Then for modules, add the module opath to the beginning.
			pkgPath, ok, err := state.getPkgPath(dir)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				break
			}
			var forTest string // only set for x tests
			isXTest := strings.HasSuffix(pkgName, "_test")
			if isXTest {
				forTest = pkgPath
				pkgPath += "_test"
			}
			id := pkgPath
			if isTestFile {
				if isXTest {
					id = fmt.Sprintf("%s [%s.test]", pkgPath, forTest)
				} else {
					id = fmt.Sprintf("%s [%s.test]", pkgPath, pkgPath)
				}
			}
			if pkg != nil {
				// TODO(rstambler): We should change the package's path and ID
				// here. The only issue is that this messes with the roots.
			} else {
				// Try to reclaim a package with the same ID, if it exists in the response.
				for _, p := range response.dr.Packages {
					if reclaimPackage(p, id, opath, contents) {
						pkg = p
						break
					}
				}
				// Otherwise, create a new package.
				if pkg == nil {
					pkg = &Package{
						PkgPath: pkgPath,
						ID:      id,
						Name:    pkgName,
						Imports: make(map[string]*Package),
					}
					response.addPackage(pkg)
					havePkgs[pkg.PkgPath] = id
					// Add the production package's sources for a test variant.
					if isTestFile && !isXTest && testVariantOf != nil {
						pkg.GoFiles = append(pkg.GoFiles, testVariantOf.GoFiles...)
						pkg.CompiledGoFiles = append(pkg.CompiledGoFiles, testVariantOf.CompiledGoFiles...)
						// Add the package under test and its imports to the test variant.
						pkg.forTest = testVariantOf.PkgPath
						for k, v := range testVariantOf.Imports {
							pkg.Imports[k] = &Package{ID: v.ID}
						}
					}
					if isXTest {
						pkg.forTest = forTest
					}
				}
			}
		}
		if !fileExists {
			pkg.GoFiles = append(pkg.GoFiles, opath)
			// TODO(matloob): Adding the file to CompiledGoFiles can exhibit the wrong behavior
			// if the file will be ignored due to its build tags.
			pkg.CompiledGoFiles = append(pkg.CompiledGoFiles, opath)
			modifiedPkgsSet[pkg.ID] = true
		}
		imports, err := extractImports(opath, contents)
		if err != nil {
			// Let the parser or type checker report errors later.
			continue
		}
		for _, imp := range imports {
			// TODO(rstambler): If the package is an x test and the import has
			// a test variant, make sure to replace it.
			if _, found := pkg.Imports[imp]; found {
				continue
			}
			overlayAddsImports = true
			id, ok := havePkgs[imp]
			if !ok {
				var err error
				id, err = state.resolveImport(dir, imp)
				if err != nil {
					return nil, nil, err
				}
			}
			pkg.Imports[imp] = &Package{ID: id}
			// Add dependencies to the non-test variant version of this package as well.
			if testVariantOf != nil {
				testVariantOf.Imports[imp] = &Package{ID: id}
			}
		}
	}

	// toPkgPath guesses the package path given the id.
	toPkgPath := func(sourceDir, id string) (string, error) {
		if i := strings.IndexByte(id, ' '); i >= 0 {
			return state.resolveImport(sourceDir, id[:i])
		}
		return state.resolveImport(sourceDir, id)
	}

	// Now that new packages have been created, do another pass to determine
	// the new set of missing packages.
	for _, pkg := range response.dr.Packages {
		for _, imp := range pkg.Imports {
			if len(pkg.GoFiles) == 0 {
				return nil, nil, fmt.Errorf("cannot resolve imports for package %q with no Go files", pkg.PkgPath)
			}
			pkgPath, err := toPkgPath(filepath.Dir(pkg.GoFiles[0]), imp.ID)
			if err != nil {
				return nil, nil, err
			}
			if _, ok := havePkgs[pkgPath]; !ok {
				needPkgsSet[pkgPath] = true
			}
		}
	}

	if overlayAddsImports {
		needPkgs = make([]string, 0, len(needPkgsSet))
		for pkg := range needPkgsSet {
			needPkgs = append(needPkgs, pkg)
		}
	}
	modifiedPkgs = make([]string, 0, len(modifiedPkgsSet))
	for pkg := range modifiedPkgsSet {
		modifiedPkgs = append(modifiedPkgs, pkg)
	}
	return modifiedPkgs, needPkgs, err
}

// resolveImport finds the ID of a package given its import path.
// In particular, it will find the right vendored copy when in GOPATH mode.
func (state *golistState) resolveImport(sourceDir, importPath string) (string, error) {
	env, err := state.getEnv()
	if err != nil {
		return "", err
	}
	if env["GOMOD"] != "" {
		return importPath, nil
	}

	searchDir := sourceDir
	for {
		vendorDir := filepath.Join(searchDir, "vendor")
		exists, ok := state.vendorDirs[vendorDir]
		if !ok {
			info, err := os.Stat(vendorDir)
			exists = err == nil && info.IsDir()
			state.vendorDirs[vendorDir] = exists
		}

		if exists {
			vendoredPath := filepath.Join(vendorDir, importPath)
			if info, err := os.Stat(vendoredPath); err == nil && info.IsDir() {
				// We should probably check for .go files here, but shame on anyone who fools us.
				path, ok, err := state.getPkgPath(vendoredPath)
				if err != nil {
					return "", err
				}
				if ok {
					return path, nil
				}
			}
		}

		// We know we've hit the top of the filesystem when we Dir / and get /,
		// or C:\ and get C:\, etc.
		next := filepath.Dir(searchDir)
		if next == searchDir {
			break
		}
		searchDir = next
	}
	return importPath, nil
}

func hasTestFiles(p *Package) bool {
	for _, f := range p.GoFiles {
		if strings.HasSuffix(f, "_test.go") {
			return true
		}
	}
	return false
}

// determineRootDirs returns a mapping from absolute directories that could
// contain code to their corresponding import path prefixes.
func (state *golistState) determineRootDirs() (map[string]string, error) {
	env, err := state.getEnv()
	if err != nil {
		return nil, err
	}
	if env["GOMOD"] != "" {
		state.rootsOnce.Do(func() {
			state.rootDirs, state.rootDirsError = state.determineRootDirsModules()
		})
	} else {
		state.rootsOnce.Do(func() {
			state.rootDirs, state.rootDirsError = state.determineRootDirsGOPATH()
		})
	}
	return state.rootDirs, state.rootDirsError
}

func (state *golistState) determineRootDirsModules() (map[string]string, error) {
	// List all of the modules--the first will be the directory for the main
	// module. Any replaced modules will also need to be treated as roots.
	// Editing files in the module cache isn't a great idea, so we don't
	// plan to ever support that.
	out, err := state.invokeGo("list", "-m", "-json", "all")
	if err != nil {
		// 'go list all' will fail if we're outside of a module and
		// GO111MODULE=on. Try falling back without 'all'.
		var innerErr error
		out, innerErr = state.invokeGo("list", "-m", "-json")
		if innerErr != nil {
			return nil, err
		}
	}
	roots := map[string]string{}
	modules := map[string]string{}
	var i int
	for dec := json.NewDecoder(out); dec.More(); {
		mod := new(gocommand.ModuleJSON)
		if err := dec.Decode(mod); err != nil {
			return nil, err
		}
		if mod.Dir != "" && mod.Path != "" {
			// This is a valid module; add it to the map.
			absDir, err := filepath.Abs(mod.Dir)
			if err != nil {
				return nil, err
			}
			modules[absDir] = mod.Path
			// The first result is the main module.
			if i == 0 || mod.Replace != nil && mod.Replace.Path != "" {
				roots[absDir] = mod.Path
			}
		}
		i++
	}
	return roots, nil
}

func (state *golistState) determineRootDirsGOPATH() (map[string]string, error) {
	m := map[string]string{}
	for _, dir := range filepath.SplitList(state.mustGetEnv()["GOPATH"]) {
		absDir, err := filepath.Abs(dir)
		if err != nil {
			return nil, err
		}
		m[filepath.Join(absDir, "src")] = ""
	}
	return m, nil
}

func extractImports(filename string, contents []byte) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), filename, contents, parser.ImportsOnly) // TODO(matloob): reuse fileset?
	if err != nil {
		return nil, err
	}
	var res []string
	for _, imp := range f.Imports {
		quotedPath := imp.Path.Value
		path, err := strconv.Unquote(quotedPath)
		if err != nil {
			return nil, err
		}
		res = append(res, path)
	}
	return res, nil
}

// reclaimPackage attempts to reuse a package that failed to load in an overlay.
//
// If the package has errors and has no Name, GoFiles, or Imports,
// then it's possible that it doesn't yet exist on disk.
func reclaimPackage(pkg *Package, id string, filename string, contents []byte) bool {
	// TODO(rstambler): Check the message of the actual error?
	// It differs between $GOPATH and module mode.
	if pkg.ID != id {
		return false
	}
	if len(pkg.Errors) != 1 {
		return false
	}
	if pkg.Name != "" || pkg.ExportFile != "" {
		return false
	}
	if len(pkg.GoFiles) > 0 || len(pkg.CompiledGoFiles) > 0 || len(pkg.OtherFiles) > 0 {
		return false
	}
	if len(pkg.Imports) > 0 {
		return false
	}
	pkgName, ok := extractPackageName(filename, contents)
	if !ok {
		return false
	}
	pkg.Name = pkgName
	pkg.Errors = nil
	return true
}

func extractPackageName(filename string, contents []byte) (string, bool) {
	// TODO(rstambler): Check the message of the actual error?
	// It differs between $GOPATH and module mode.
	f, err := parser.ParseFile(token.NewFileSet(), filename, contents, parser.PackageClauseOnly) // TODO(matloob): reuse fileset?
	if err != nil {
		return "", false
	}
	return f.Name.Name, true
}

// commonDir returns the directory that all files are in, "" if files is empty,
// or an error if they aren't in the same directory.
func commonDir(files []string) (string, error) {
	seen := make(map[string]bool)
	for _, f := range files {
		seen[filepath.Dir(f)] = true
	}
	if len(seen) > 1 {
		return "", fmt.Errorf("files (%v) are in more than one directory: %v", files, seen)
	}
	for k := range seen {
		// seen has only one element; return it.
		return k, nil
	}
	return "", nil // no files
}

// It is possible that the files in the disk directory dir have a different package
// name from newName, which is deduced from the overlays. If they all have a different
// package name, and they all have the same package name, then that name becomes
// the package name.
// It returns true if it changes the package name, false otherwise.
func maybeFixPackageName(newName string, isTestFile bool, pkgsOfDir []*Package) {
	names := make(map[string]int)
	for _, p := range pkgsOfDir {
		names[p.Name]++
	}
	if len(names) != 1 {
		// some files are in different packages
		return
	}
	var oldName string
	for k := range names {
		oldName = k
	}
	if newName == oldName {
		return
	}
	// We might have a case where all of the package names in the directory are
	// the same, but the overlay file is for an x test, which belongs to its
	// own package. If the x test does not yet exist on disk, we may not yet
	// have its package name on disk, but we should not rename the packages.
	//
	// We use a heuristic to determine if this file belongs to an x test:
	// The test file should have a package name whose package name has a _test
	// suffix or looks like "newName_test".
	maybeXTest := strings.HasPrefix(oldName+"_test", newName) || strings.HasSuffix(newName, "_test")
	if isTestFile && maybeXTest {
		return
	}
	for _, p := range pkgsOfDir {
		p.Name = newName
	}
}

// This function is copy-pasted from
// https://github.com/golang/go/blob/9706f510a5e2754595d716bd64be8375997311fb/src/cmd/go/internal/search/search.go#L360.
// It should be deleted when we remove support for overlays from go/packages.
//
// NOTE: This does not handle any ./... or ./ style queries, as this function
// doesn't know the working directory.
//
// matchPattern(pattern)(name) reports whether
// name matches pattern. Pattern is a limited glob
// pattern in which '...' means 'any string' and there
// is no other special syntax.
// Unfortunately, there are two special cases. Quoting "go help packages":
//
// First, /... at the end of the pattern can match an empty string,
// so that net/... matches both net and packages in its subdirectories, like net/http.
// Second, any slash-separated pattern element containing a wildcard never
// participates in a match of the "vendor" element in the path of a vendored
// package, so that ./... does not match packages in subdirectories of
// ./vendor or ./mycode/vendor, but ./vendor/... and ./mycode/vendor/... do.
// Note, however, that a directory named vendor that itself contains code
// is not a vendored package: cmd/vendor would be a command named vendor,
// and the pattern cmd/... matches it.
func matchPattern(pattern string) func(name string) bool {
	// Convert pattern to regular expression.
	// The strategy for the trailing /... is to nest it in an explicit ? expression.
	// The strategy for the vendor exclusion is to change the unmatchable
	// vendor strings to a disallowed code point (vendorChar) and to use
	// "(anything but that codepoint)*" as the implementation of the ... wildcard.
	// This is a bit complicated but the obvious alternative,
	// namely a hand-written search like in most shell glob matchers,
	// is too easy to make accidentally exponential.
	// Using package regexp guarantees linear-time matching.

	const vendorChar = "\x00"

	if strings.Contains(pattern, vendorChar) {
		return func(name string) bool { return false }
	}

	re := regexp.QuoteMeta(pattern)
	re = replaceVendor(re, vendorChar)
	switch {
	case strings.HasSuffix(re, `/`+vendorChar+`/\.\.\.`):
		re = strings.TrimSuffix(re, `/`+vendorChar+`/\.\.\.`) + `(/vendor|/` + vendorChar + `/\.\.\.)`
	case re == vendorChar+`/\.\.\.`:
		re = `(/vendor|/` + vendorChar + `/\.\.\.)`
	case strings.HasSuffix(re, `/\.\.\.`):
		re = strings.TrimSuffix(re, `/\.\.\.`) + `(/\.\.\.)?`
	}
	re = strings.ReplaceAll(re, `\.\.\.`, `[^`+vendorChar+`]*`)

	reg := regexp.MustCompile(`^` + re + `$`)

	return func(name string) bool {
		if strings.Contains(name, vendorChar) {
			return false
		}
		return reg.MatchString(replaceVendor(name, vendorChar))
	}
}

// replaceVendor returns the result of replacing
// non-trailing vendor path elements in x with repl.
func replaceVendor(x, repl string) string {
	if !strings.Contains(x, "vendor") {
		return x
	}
	elem := strings.Split(x, "/")
	for i := 0; i < len(elem)-1; i++ {
		if elem[i] == "vendor" {
			elem[i] = repl
		}
	}
	return strings.Join(elem, "/")
}
