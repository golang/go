package imports

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/module"
)

// ModuleResolver implements resolver for modules using the go command as little
// as feasible.
type ModuleResolver struct {
	env *ProcessEnv

	Initialized   bool
	Main          *ModuleJSON
	ModsByModPath []*ModuleJSON // All modules, ordered by # of path components in module Path...
	ModsByDir     []*ModuleJSON // ...or Dir.

	// ModCachePkgs contains the canonicalized importPath and directory of packages
	// in the module cache. Keyed by absolute directory.
	ModCachePkgs map[string]*pkg

	// moduleCacheInfo stores information about the module cache.
	moduleCacheInfo *moduleCacheInfo
}

type ModuleJSON struct {
	Path     string           // module path
	Version  string           // module version
	Versions []string         // available module versions (with -versions)
	Replace  *ModuleJSON      // replaced by this module
	Time     *time.Time       // time version was created
	Update   *ModuleJSON      // available update, if any (with -u)
	Main     bool             // is this the main module?
	Indirect bool             // is this module only an indirect dependency of main module?
	Dir      string           // directory holding files for this module, if any
	GoMod    string           // path to go.mod file for this module, if any
	Error    *ModuleErrorJSON // error loading module
}

type ModuleErrorJSON struct {
	Err string // the error itself
}

func (r *ModuleResolver) init() error {
	if r.Initialized {
		return nil
	}
	stdout, err := r.env.invokeGo("list", "-m", "-json", "...")
	if err != nil {
		return err
	}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &ModuleJSON{}
		if err := dec.Decode(mod); err != nil {
			return err
		}
		if mod.Dir == "" {
			if r.env.Debug {
				r.env.Logf("module %v has not been downloaded and will be ignored", mod.Path)
			}
			// Can't do anything with a module that's not downloaded.
			continue
		}
		r.ModsByModPath = append(r.ModsByModPath, mod)
		r.ModsByDir = append(r.ModsByDir, mod)
		if mod.Main {
			r.Main = mod
		}
	}

	sort.Slice(r.ModsByModPath, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.ModsByModPath[x].Path, "/")
		}
		return count(j) < count(i) // descending order
	})
	sort.Slice(r.ModsByDir, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.ModsByDir[x].Dir, "/")
		}
		return count(j) < count(i) // descending order
	})

	r.ModCachePkgs = make(map[string]*pkg)
	if r.moduleCacheInfo == nil {
		r.moduleCacheInfo = &moduleCacheInfo{
			modCacheDirInfo: make(map[string]*directoryPackageInfo),
		}
	}

	r.Initialized = true
	return nil
}

// findPackage returns the module and directory that contains the package at
// the given import path, or returns nil, "" if no module is in scope.
func (r *ModuleResolver) findPackage(importPath string) (*ModuleJSON, string) {
	for _, m := range r.ModsByModPath {
		if !strings.HasPrefix(importPath, m.Path) {
			continue
		}
		pathInModule := importPath[len(m.Path):]
		pkgDir := filepath.Join(m.Dir, pathInModule)
		if dirIsNestedModule(pkgDir, m) {
			continue
		}

		pkgFiles, err := ioutil.ReadDir(pkgDir)
		if err != nil {
			continue
		}

		// A module only contains a package if it has buildable go
		// files in that directory. If not, it could be provided by an
		// outer module. See #29736.
		for _, fi := range pkgFiles {
			if ok, _ := r.env.buildContext().MatchFile(pkgDir, fi.Name()); ok {
				return m, pkgDir
			}
		}
	}
	return nil, ""
}

// findModuleByDir returns the module that contains dir, or nil if no such
// module is in scope.
func (r *ModuleResolver) findModuleByDir(dir string) *ModuleJSON {
	// This is quite tricky and may not be correct. dir could be:
	// - a package in the main module.
	// - a replace target underneath the main module's directory.
	//    - a nested module in the above.
	// - a replace target somewhere totally random.
	//    - a nested module in the above.
	// - in the mod cache.
	// - in /vendor/ in -mod=vendor mode.
	//    - nested module? Dunno.
	// Rumor has it that replace targets cannot contain other replace targets.
	for _, m := range r.ModsByDir {
		if !strings.HasPrefix(dir, m.Dir) {
			continue
		}

		if dirIsNestedModule(dir, m) {
			continue
		}

		return m
	}
	return nil
}

// dirIsNestedModule reports if dir is contained in a nested module underneath
// mod, not actually in mod.
func dirIsNestedModule(dir string, mod *ModuleJSON) bool {
	if !strings.HasPrefix(dir, mod.Dir) {
		return false
	}
	mf := findModFile(dir)
	if mf == "" {
		return false
	}
	return filepath.Dir(mf) != mod.Dir
}

func findModFile(dir string) string {
	for {
		f := filepath.Join(dir, "go.mod")
		info, err := os.Stat(f)
		if err == nil && !info.IsDir() {
			return f
		}
		d := filepath.Dir(dir)
		if len(d) >= len(dir) {
			return "" // reached top of file system, no go.mod
		}
		dir = d
	}
}

func (r *ModuleResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	if err := r.init(); err != nil {
		return nil, err
	}
	names := map[string]string{}
	for _, path := range importPaths {
		_, packageDir := r.findPackage(path)
		if packageDir == "" {
			continue
		}
		name, err := packageDirToName(packageDir)
		if err != nil {
			continue
		}
		names[path] = name
	}
	return names, nil
}

func (r *ModuleResolver) scan(_ references) ([]*pkg, error) {
	if err := r.init(); err != nil {
		return nil, err
	}

	// Walk GOROOT, GOPATH/pkg/mod, and the main module.
	roots := []gopathwalk.Root{
		{filepath.Join(r.env.GOROOT, "/src"), gopathwalk.RootGOROOT},
	}
	if r.Main != nil {
		roots = append(roots, gopathwalk.Root{r.Main.Dir, gopathwalk.RootCurrentModule})
	}
	for _, p := range filepath.SplitList(r.env.GOPATH) {
		roots = append(roots, gopathwalk.Root{filepath.Join(p, "/pkg/mod"), gopathwalk.RootModuleCache})
	}

	// Walk replace targets, just in case they're not in any of the above.
	for _, mod := range r.ModsByModPath {
		if mod.Replace != nil {
			roots = append(roots, gopathwalk.Root{mod.Dir, gopathwalk.RootOther})
		}
	}

	var result []*pkg
	dupCheck := make(map[string]bool)
	var mu sync.Mutex

	gopathwalk.Walk(roots, func(root gopathwalk.Root, dir string) {
		mu.Lock()
		defer mu.Unlock()

		if _, dup := dupCheck[dir]; dup {
			return
		}

		dupCheck[dir] = true

		absDir := dir
		// Packages in the module cache are immutable. If we have
		// already seen this package on a previous scan of the module
		// cache, return that result.
		if p, ok := r.ModCachePkgs[absDir]; ok {
			result = append(result, p)
			return
		}

		info, ok := r.moduleCacheInfo.Load(dir)
		if !ok {
			var err error
			info, err = r.scanDirForPackage(root, dir)
			if err != nil {
				return
			}
			if root.Type == gopathwalk.RootModuleCache {
				r.moduleCacheInfo.Store(dir, info)
			}
		}

		if info.status < directoryScanned ||
			(info.status == directoryScanned && info.err != nil) {
			return
		}

		// The rest of this function canonicalizes the packages using the results
		// of initializing the resolver from 'go list -m'.
		importPath := info.nonCanonicalImportPath

		// Check if the directory is underneath a module that's in scope.
		if mod := r.findModuleByDir(dir); mod != nil {
			// It is. If dir is the target of a replace directive,
			// our guessed import path is wrong. Use the real one.
			if mod.Dir == dir {
				importPath = mod.Path
			} else {
				dirInMod := dir[len(mod.Dir)+len("/"):]
				importPath = path.Join(mod.Path, filepath.ToSlash(dirInMod))
			}
		} else if info.needsReplace {
			// This package needed a replace target we don't have.
			return
		}

		// We may have discovered a package that has a different version
		// in scope already. Canonicalize to that one if possible.
		if _, canonicalDir := r.findPackage(importPath); canonicalDir != "" {
			dir = canonicalDir
		}

		res := &pkg{
			importPathShort: VendorlessPath(importPath),
			dir:             dir,
		}

		if root.Type == gopathwalk.RootModuleCache {
			// Save the results of processing this directory.
			// This needs to be invalidated when the results of
			// 'go list -m' would change, as the directory and
			// importPath in this map depend on those results.
			r.ModCachePkgs[absDir] = res
		}

		result = append(result, res)
	}, gopathwalk.Options{Debug: r.env.Debug, ModulesEnabled: true})
	return result, nil
}

func (r *ModuleResolver) loadExports(ctx context.Context, expectPackage string, pkg *pkg) (map[string]bool, error) {
	if err := r.init(); err != nil {
		return nil, err
	}
	return loadExportsFromFiles(ctx, r.env, expectPackage, pkg.dir)
}

func (r *ModuleResolver) scanDirForPackage(root gopathwalk.Root, dir string) (directoryPackageInfo, error) {
	subdir := ""
	if dir != root.Path {
		subdir = dir[len(root.Path)+len("/"):]
	}
	importPath := filepath.ToSlash(subdir)
	if strings.HasPrefix(importPath, "vendor/") {
		// Ignore vendor dirs. If -mod=vendor is on, then things
		// should mostly just work, but when it's not vendor/
		// is a mess. There's no easy way to tell if it's on.
		// We can still find things in the mod cache and
		// map them into /vendor when -mod=vendor is on.
		return directoryPackageInfo{}, fmt.Errorf("vendor directory")
	}
	switch root.Type {
	case gopathwalk.RootCurrentModule:
		importPath = path.Join(r.Main.Path, filepath.ToSlash(subdir))
	case gopathwalk.RootModuleCache:
		matches := modCacheRegexp.FindStringSubmatch(subdir)
		modPath, err := module.DecodePath(filepath.ToSlash(matches[1]))
		if err != nil {
			if r.env.Debug {
				r.env.Logf("decoding module cache path %q: %v", subdir, err)
			}
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("decoding module cache path %q: %v", subdir, err),
			}, nil
		}
		importPath = path.Join(modPath, filepath.ToSlash(matches[3]))
	case gopathwalk.RootGOROOT:
		importPath = subdir
	}

	// Check that this package is not obviously impossible to import.
	var modFile string
	switch root.Type {
	case gopathwalk.RootModuleCache:
		matches := modCacheRegexp.FindStringSubmatch(subdir)
		index := strings.Index(dir, matches[1]+"@"+matches[2])
		modFile = filepath.Join(dir[:index], matches[1]+"@"+matches[2], "go.mod")
	default:
		modFile = findModFile(dir)
	}

	var needsReplace bool
	modBytes, err := ioutil.ReadFile(modFile)
	if err == nil && !strings.HasPrefix(importPath, modulePath(modBytes)) {
		// The module's declared path does not match
		// its expected path. It probably needs a
		// replace directive we don't have.
		needsReplace = true
	}

	return directoryPackageInfo{
		status:                 directoryScanned,
		dir:                    dir,
		nonCanonicalImportPath: importPath,
		needsReplace:           needsReplace,
	}, nil
}

// modCacheRegexp splits a path in a module cache into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

var (
	slashSlash = []byte("//")
	moduleStr  = []byte("module")
)

// modulePath returns the module path from the gomod file text.
// If it cannot find a module path, it returns an empty string.
// It is tolerant of unrelated problems in the go.mod file.
//
// Copied from cmd/go/internal/modfile.
func modulePath(mod []byte) string {
	for len(mod) > 0 {
		line := mod
		mod = nil
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, mod = line[:i], line[i+1:]
		}
		if i := bytes.Index(line, slashSlash); i >= 0 {
			line = line[:i]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, moduleStr) {
			continue
		}
		line = line[len(moduleStr):]
		n := len(line)
		line = bytes.TrimSpace(line)
		if len(line) == n || len(line) == 0 {
			continue
		}

		if line[0] == '"' || line[0] == '`' {
			p, err := strconv.Unquote(string(line))
			if err != nil {
				return "" // malformed quoted string or multiline module path
			}
			return p
		}

		return string(line)
	}
	return "" // missing module path
}
