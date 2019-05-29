package packages

import (
	"bytes"
	"encoding/json"
	"go/parser"
	"go/token"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// processGolistOverlay provides rudimentary support for adding
// files that don't exist on disk to an overlay. The results can be
// sometimes incorrect.
// TODO(matloob): Handle unsupported cases, including the following:
// - test files
// - adding test and non-test files to test variants of packages
// - determining the correct package to add given a new import path
// - creating packages that don't exist
func processGolistOverlay(cfg *Config, response *driverResponse) (modifiedPkgs, needPkgs []string, err error) {
	havePkgs := make(map[string]string) // importPath -> non-test package ID
	needPkgsSet := make(map[string]bool)
	modifiedPkgsSet := make(map[string]bool)

	for _, pkg := range response.Packages {
		// This is an approximation of import path to id. This can be
		// wrong for tests, vendored packages, and a number of other cases.
		havePkgs[pkg.PkgPath] = pkg.ID
	}

	var rootDirs map[string]string
	var onceGetRootDirs sync.Once

	for opath, contents := range cfg.Overlay {
		base := filepath.Base(opath)
		if strings.HasSuffix(opath, "_test.go") {
			// Overlays don't support adding new test files yet.
			// TODO(matloob): support adding new test files.
			continue
		}
		dir := filepath.Dir(opath)
		var pkg *Package
		var fileExists bool
		for _, p := range response.Packages {
			for _, f := range p.GoFiles {
				if !sameFile(filepath.Dir(f), dir) {
					continue
				}
				pkg = p
				if filepath.Base(f) == base {
					fileExists = true
				}
			}
		}
		// The overlay could have included an entirely new package.
		if pkg == nil {
			onceGetRootDirs.Do(func() {
				rootDirs = determineRootDirs(cfg)
			})
			// Try to find the module or gopath dir the file is contained in.
			// Then for modules, add the module opath to the beginning.
			var pkgPath string
			for rdir, rpath := range rootDirs {
				// TODO(matloob): This doesn't properly handle symlinks.
				r, err := filepath.Rel(rdir, dir)
				if err != nil {
					continue
				}
				pkgPath = filepath.ToSlash(r)
				if rpath != "" {
					pkgPath = path.Join(rpath, pkgPath)
				}
				// We only create one new package even it can belong in multiple modules or GOPATH entries.
				// This is okay because tools (such as the LSP) that use overlays will recompute the overlay
				// once the file is saved, and golist will do the right thing.
				// TODO(matloob): Implement module tiebreaking?
				break
			}
			if pkgPath == "" {
				continue
			}
			pkgName, ok := extractPackageName(opath, contents)
			if !ok {
				continue
			}
			id := pkgPath
			// Try to reclaim a package with the same id if it exists in the response.
			for _, p := range response.Packages {
				if reclaimPackage(p, id, opath, contents) {
					pkg = p
					break
				}
			}
			// Otherwise, create a new package
			if pkg == nil {
				pkg = &Package{PkgPath: pkgPath, ID: id, Name: pkgName, Imports: make(map[string]*Package)}
				// TODO(matloob): Is it okay to amend response.Packages this way?
				response.Packages = append(response.Packages, pkg)
				havePkgs[pkg.PkgPath] = id
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
			_, found := pkg.Imports[imp]
			if !found {
				// TODO(matloob): Handle cases when the following block isn't correct.
				// These include imports of test variants, imports of vendored packages, etc.
				id, ok := havePkgs[imp]
				if !ok {
					id = imp
				}
				pkg.Imports[imp] = &Package{ID: id}
			}
		}
		continue
	}

	// toPkgPath tries to guess the package path given the id.
	// This isn't always correct -- it's certainly wrong for
	// vendored packages' paths.
	toPkgPath := func(id string) string {
		// TODO(matloob): Handle vendor paths.
		i := strings.IndexByte(id, ' ')
		if i >= 0 {
			return id[:i]
		}
		return id
	}

	// Do another pass now that new packages have been created to determine the
	// set of missing packages.
	for _, pkg := range response.Packages {
		for _, imp := range pkg.Imports {
			pkgPath := toPkgPath(imp.ID)
			if _, ok := havePkgs[pkgPath]; !ok {
				needPkgsSet[pkgPath] = true
			}
		}
	}

	needPkgs = make([]string, 0, len(needPkgsSet))
	for pkg := range needPkgsSet {
		needPkgs = append(needPkgs, pkg)
	}
	modifiedPkgs = make([]string, 0, len(modifiedPkgsSet))
	for pkg := range modifiedPkgsSet {
		modifiedPkgs = append(modifiedPkgs, pkg)
	}
	return modifiedPkgs, needPkgs, err
}

// determineRootDirs returns a mapping from directories code can be contained in to the
// corresponding import path prefixes of those directories.
// Its result is used to try to determine the import path for a package containing
// an overlay file.
func determineRootDirs(cfg *Config) map[string]string {
	// Assume modules first:
	out, err := invokeGo(cfg, "list", "-m", "-json", "all")
	if err != nil {
		return determineRootDirsGOPATH(cfg)
	}
	m := map[string]string{}
	type jsonMod struct{ Path, Dir string }
	for dec := json.NewDecoder(out); dec.More(); {
		mod := new(jsonMod)
		if err := dec.Decode(mod); err != nil {
			return m // Give up and return an empty map. Package won't be found for overlay.
		}
		if mod.Dir != "" && mod.Path != "" {
			// This is a valid module; add it to the map.
			m[mod.Dir] = mod.Path
		}
	}
	return m
}

func determineRootDirsGOPATH(cfg *Config) map[string]string {
	m := map[string]string{}
	out, err := invokeGo(cfg, "env", "GOPATH")
	if err != nil {
		// Could not determine root dir mapping. Everything is best-effort, so just return an empty map.
		// When we try to find the import path for a directory, there will be no root-dir match and
		// we'll give up.
		return m
	}
	for _, p := range filepath.SplitList(string(bytes.TrimSpace(out.Bytes()))) {
		m[filepath.Join(p, "src")] = ""
	}
	return m
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
