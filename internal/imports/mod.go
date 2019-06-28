package imports

import (
	"bytes"
	"encoding/json"
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

// moduleResolver implements resolver for modules using the go command as little
// as feasible.
type moduleResolver struct {
	env *ProcessEnv

	initialized   bool
	main          *moduleJSON
	modsByModPath []*moduleJSON // All modules, ordered by # of path components in module Path...
	modsByDir     []*moduleJSON // ...or Dir.
}

type moduleJSON struct {
	Path     string           // module path
	Version  string           // module version
	Versions []string         // available module versions (with -versions)
	Replace  *moduleJSON      // replaced by this module
	Time     *time.Time       // time version was created
	Update   *moduleJSON      // available update, if any (with -u)
	Main     bool             // is this the main module?
	Indirect bool             // is this module only an indirect dependency of main module?
	Dir      string           // directory holding files for this module, if any
	GoMod    string           // path to go.mod file for this module, if any
	Error    *moduleErrorJSON // error loading module
}

type moduleErrorJSON struct {
	Err string // the error itself
}

func (r *moduleResolver) init() error {
	if r.initialized {
		return nil
	}
	stdout, err := r.env.invokeGo("list", "-m", "-json", "...")
	if err != nil {
		return err
	}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &moduleJSON{}
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
		r.modsByModPath = append(r.modsByModPath, mod)
		r.modsByDir = append(r.modsByDir, mod)
		if mod.Main {
			r.main = mod
		}
	}

	sort.Slice(r.modsByModPath, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.modsByModPath[x].Path, "/")
		}
		return count(j) < count(i) // descending order
	})
	sort.Slice(r.modsByDir, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.modsByDir[x].Dir, "/")
		}
		return count(j) < count(i) // descending order
	})

	r.initialized = true
	return nil
}

// findPackage returns the module and directory that contains the package at
// the given import path, or returns nil, "" if no module is in scope.
func (r *moduleResolver) findPackage(importPath string) (*moduleJSON, string) {
	for _, m := range r.modsByModPath {
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
func (r *moduleResolver) findModuleByDir(dir string) *moduleJSON {
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
	for _, m := range r.modsByDir {
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
func dirIsNestedModule(dir string, mod *moduleJSON) bool {
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

func (r *moduleResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
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

func (r *moduleResolver) scan(_ references) ([]*pkg, error) {
	if err := r.init(); err != nil {
		return nil, err
	}

	// Walk GOROOT, GOPATH/pkg/mod, and the main module.
	roots := []gopathwalk.Root{
		{filepath.Join(r.env.GOROOT, "/src"), gopathwalk.RootGOROOT},
	}
	if r.main != nil {
		roots = append(roots, gopathwalk.Root{r.main.Dir, gopathwalk.RootCurrentModule})
	}
	for _, p := range filepath.SplitList(r.env.GOPATH) {
		roots = append(roots, gopathwalk.Root{filepath.Join(p, "/pkg/mod"), gopathwalk.RootModuleCache})
	}

	// Walk replace targets, just in case they're not in any of the above.
	for _, mod := range r.modsByModPath {
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
			return
		}
		switch root.Type {
		case gopathwalk.RootCurrentModule:
			importPath = path.Join(r.main.Path, filepath.ToSlash(subdir))
		case gopathwalk.RootModuleCache:
			matches := modCacheRegexp.FindStringSubmatch(subdir)
			modPath, err := module.DecodePath(filepath.ToSlash(matches[1]))
			if err != nil {
				if r.env.Debug {
					r.env.Logf("decoding module cache path %q: %v", subdir, err)
				}
				return
			}
			importPath = path.Join(modPath, filepath.ToSlash(matches[3]))
		case gopathwalk.RootGOROOT:
			importPath = subdir
		}

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
		} else {
			// The package is in an unknown module. Check that it's
			// not obviously impossible to import.
			var modFile string
			switch root.Type {
			case gopathwalk.RootModuleCache:
				matches := modCacheRegexp.FindStringSubmatch(subdir)
				modFile = filepath.Join(matches[1], "@", matches[2], "go.mod")
			default:
				modFile = findModFile(dir)
			}

			modBytes, err := ioutil.ReadFile(modFile)
			if err == nil && !strings.HasPrefix(importPath, modulePath(modBytes)) {
				// The module's declared path does not match
				// its expected path. It probably needs a
				// replace directive we don't have.
				return
			}
		}
		// We may have discovered a package that has a different version
		// in scope already. Canonicalize to that one if possible.
		if _, canonicalDir := r.findPackage(importPath); canonicalDir != "" {
			dir = canonicalDir
		}

		result = append(result, &pkg{
			importPathShort: VendorlessPath(importPath),
			dir:             dir,
		})
	}, gopathwalk.Options{Debug: r.env.Debug, ModulesEnabled: true})
	return result, nil
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
