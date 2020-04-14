package packages

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
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
		x := commonDir(pkg.GoFiles)
		if x != "" {
			pkgOfDir[x] = append(pkgOfDir[x], pkg)
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
		// if all the overlay files belong to a different package, change the package
		// name to that package. Otherwise leave it alone; there will be an error message.
		maybeFixPackageName(pkgName, pkgOfDir, dir)
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
				}
				// We must have already seen the package of which this is a test variant.
				if pkg != nil && p != pkg && pkg.PkgPath == p.PkgPath {
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
		// The overlay could have included an entirely new package.
		if pkg == nil {
			// Try to find the module or gopath dir the file is contained in.
			// Then for modules, add the module opath to the beginning.
			pkgPath, ok, err := state.getPkgPath(dir)
			if err != nil {
				return nil, nil, err
			}
			if !ok {
				break
			}
			isXTest := strings.HasSuffix(pkgName, "_test")
			if isXTest {
				pkgPath += "_test"
			}
			id := pkgPath
			if isTestFile && !isXTest {
				id = fmt.Sprintf("%s [%s.test]", pkgPath, pkgPath)
			}
			// Try to reclaim a package with the same id if it exists in the response.
			for _, p := range response.dr.Packages {
				if reclaimPackage(p, id, opath, contents) {
					pkg = p
					break
				}
			}
			// Otherwise, create a new package
			if pkg == nil {
				pkg = &Package{PkgPath: pkgPath, ID: id, Name: pkgName, Imports: make(map[string]*Package)}
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

// resolveImport finds the the ID of a package given its import path.
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
	// This will only return the root directory for the main module.
	// For now we only support overlays in main modules.
	// Editing files in the module cache isn't a great idea, so we don't
	// plan to ever support that, but editing files in replaced modules
	// is something we may want to support. To do that, we'll want to
	// do a go list -m to determine the replaced module's module path and
	// directory, and then a go list -m {{with .Replace}}{{.Dir}}{{end}} <replaced module's path>
	// from the main module to determine if that module is actually a replacement.
	// See bcmills's comment here: https://github.com/golang/go/issues/37629#issuecomment-594179751
	// for more information.
	out, err := state.invokeGo("list", "-m", "-json")
	if err != nil {
		return nil, err
	}
	m := map[string]string{}
	type jsonMod struct{ Path, Dir string }
	for dec := json.NewDecoder(out); dec.More(); {
		mod := new(jsonMod)
		if err := dec.Decode(mod); err != nil {
			return nil, err
		}
		if mod.Dir != "" && mod.Path != "" {
			// This is a valid module; add it to the map.
			absDir, err := filepath.Abs(mod.Dir)
			if err != nil {
				return nil, err
			}
			m[absDir] = mod.Path
		}
	}
	return m, nil
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

func commonDir(a []string) string {
	seen := make(map[string]bool)
	x := append([]string{}, a...)
	for _, f := range x {
		seen[filepath.Dir(f)] = true
	}
	if len(seen) > 1 {
		log.Fatalf("commonDir saw %v for %v", seen, x)
	}
	for k := range seen {
		// len(seen) == 1
		return k
	}
	return "" // no files
}

// It is possible that the files in the disk directory dir have a different package
// name from newName, which is deduced from the overlays. If they all have a different
// package name, and they all have the same package name, then that name becomes
// the package name.
// It returns true if it changes the package name, false otherwise.
func maybeFixPackageName(newName string, pkgOfDir map[string][]*Package, dir string) bool {
	names := make(map[string]int)
	for _, p := range pkgOfDir[dir] {
		names[p.Name]++
	}
	if len(names) != 1 {
		// some files are in different packages
		return false
	}
	oldName := ""
	for k := range names {
		oldName = k
	}
	if newName == oldName {
		return false
	}
	for _, p := range pkgOfDir[dir] {
		p.Name = newName
	}
	return true
}
