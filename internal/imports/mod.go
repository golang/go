// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/gopathwalk"
)

// ModuleResolver implements resolver for modules using the go command as little
// as feasible.
type ModuleResolver struct {
	env            *ProcessEnv
	moduleCacheDir string
	dummyVendorMod *gocommand.ModuleJSON // If vendoring is enabled, the pseudo-module that represents the /vendor directory.
	roots          []gopathwalk.Root
	scanSema       chan struct{} // scanSema prevents concurrent scans and guards scannedRoots.
	scannedRoots   map[gopathwalk.Root]bool

	initialized   bool
	mains         []*gocommand.ModuleJSON
	mainByDir     map[string]*gocommand.ModuleJSON
	modsByModPath []*gocommand.ModuleJSON // All modules, ordered by # of path components in module Path...
	modsByDir     []*gocommand.ModuleJSON // ...or Dir.

	// moduleCacheCache stores information about the module cache.
	moduleCacheCache *dirInfoCache
	otherCache       *dirInfoCache
}

func newModuleResolver(e *ProcessEnv) *ModuleResolver {
	r := &ModuleResolver{
		env:      e,
		scanSema: make(chan struct{}, 1),
	}
	r.scanSema <- struct{}{}
	return r
}

func (r *ModuleResolver) init() error {
	if r.initialized {
		return nil
	}

	goenv, err := r.env.goEnv()
	if err != nil {
		return err
	}
	inv := gocommand.Invocation{
		BuildFlags: r.env.BuildFlags,
		ModFlag:    r.env.ModFlag,
		ModFile:    r.env.ModFile,
		Env:        r.env.env(),
		Logf:       r.env.Logf,
		WorkingDir: r.env.WorkingDir,
	}

	vendorEnabled := false
	var mainModVendor *gocommand.ModuleJSON

	// Module vendor directories are ignored in workspace mode:
	// https://go.googlesource.com/proposal/+/master/design/45713-workspace.md
	if len(r.env.Env["GOWORK"]) == 0 {
		vendorEnabled, mainModVendor, err = gocommand.VendorEnabled(context.TODO(), inv, r.env.GocmdRunner)
		if err != nil {
			return err
		}
	}

	if mainModVendor != nil && vendorEnabled {
		// Vendor mode is on, so all the non-Main modules are irrelevant,
		// and we need to search /vendor for everything.
		r.mains = []*gocommand.ModuleJSON{mainModVendor}
		r.dummyVendorMod = &gocommand.ModuleJSON{
			Path: "",
			Dir:  filepath.Join(mainModVendor.Dir, "vendor"),
		}
		r.modsByModPath = []*gocommand.ModuleJSON{mainModVendor, r.dummyVendorMod}
		r.modsByDir = []*gocommand.ModuleJSON{mainModVendor, r.dummyVendorMod}
	} else {
		// Vendor mode is off, so run go list -m ... to find everything.
		err := r.initAllMods()
		// We expect an error when running outside of a module with
		// GO111MODULE=on. Other errors are fatal.
		if err != nil {
			if errMsg := err.Error(); !strings.Contains(errMsg, "working directory is not part of a module") && !strings.Contains(errMsg, "go.mod file not found") {
				return err
			}
		}
	}

	if gmc := r.env.Env["GOMODCACHE"]; gmc != "" {
		r.moduleCacheDir = gmc
	} else {
		gopaths := filepath.SplitList(goenv["GOPATH"])
		if len(gopaths) == 0 {
			return fmt.Errorf("empty GOPATH")
		}
		r.moduleCacheDir = filepath.Join(gopaths[0], "/pkg/mod")
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

	r.roots = []gopathwalk.Root{
		{Path: filepath.Join(goenv["GOROOT"], "/src"), Type: gopathwalk.RootGOROOT},
	}
	r.mainByDir = make(map[string]*gocommand.ModuleJSON)
	for _, main := range r.mains {
		r.roots = append(r.roots, gopathwalk.Root{Path: main.Dir, Type: gopathwalk.RootCurrentModule})
		r.mainByDir[main.Dir] = main
	}
	if vendorEnabled {
		r.roots = append(r.roots, gopathwalk.Root{Path: r.dummyVendorMod.Dir, Type: gopathwalk.RootOther})
	} else {
		addDep := func(mod *gocommand.ModuleJSON) {
			if mod.Replace == nil {
				// This is redundant with the cache, but we'll skip it cheaply enough.
				r.roots = append(r.roots, gopathwalk.Root{Path: mod.Dir, Type: gopathwalk.RootModuleCache})
			} else {
				r.roots = append(r.roots, gopathwalk.Root{Path: mod.Dir, Type: gopathwalk.RootOther})
			}
		}
		// Walk dependent modules before scanning the full mod cache, direct deps first.
		for _, mod := range r.modsByModPath {
			if !mod.Indirect && !mod.Main {
				addDep(mod)
			}
		}
		for _, mod := range r.modsByModPath {
			if mod.Indirect && !mod.Main {
				addDep(mod)
			}
		}
		r.roots = append(r.roots, gopathwalk.Root{Path: r.moduleCacheDir, Type: gopathwalk.RootModuleCache})
	}

	r.scannedRoots = map[gopathwalk.Root]bool{}
	if r.moduleCacheCache == nil {
		r.moduleCacheCache = &dirInfoCache{
			dirs:      map[string]*directoryPackageInfo{},
			listeners: map[*int]cacheListener{},
		}
	}
	if r.otherCache == nil {
		r.otherCache = &dirInfoCache{
			dirs:      map[string]*directoryPackageInfo{},
			listeners: map[*int]cacheListener{},
		}
	}
	r.initialized = true
	return nil
}

func (r *ModuleResolver) initAllMods() error {
	stdout, err := r.env.invokeGo(context.TODO(), "list", "-m", "-e", "-json", "...")
	if err != nil {
		return err
	}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &gocommand.ModuleJSON{}
		if err := dec.Decode(mod); err != nil {
			return err
		}
		if mod.Dir == "" {
			if r.env.Logf != nil {
				r.env.Logf("module %v has not been downloaded and will be ignored", mod.Path)
			}
			// Can't do anything with a module that's not downloaded.
			continue
		}
		// golang/go#36193: the go command doesn't always clean paths.
		mod.Dir = filepath.Clean(mod.Dir)
		r.modsByModPath = append(r.modsByModPath, mod)
		r.modsByDir = append(r.modsByDir, mod)
		if mod.Main {
			r.mains = append(r.mains, mod)
		}
	}
	return nil
}

func (r *ModuleResolver) ClearForNewScan() {
	<-r.scanSema
	r.scannedRoots = map[gopathwalk.Root]bool{}
	r.otherCache = &dirInfoCache{
		dirs:      map[string]*directoryPackageInfo{},
		listeners: map[*int]cacheListener{},
	}
	r.scanSema <- struct{}{}
}

func (r *ModuleResolver) ClearForNewMod() {
	<-r.scanSema
	*r = ModuleResolver{
		env:              r.env,
		moduleCacheCache: r.moduleCacheCache,
		otherCache:       r.otherCache,
		scanSema:         r.scanSema,
	}
	r.init()
	r.scanSema <- struct{}{}
}

// findPackage returns the module and directory that contains the package at
// the given import path, or returns nil, "" if no module is in scope.
func (r *ModuleResolver) findPackage(importPath string) (*gocommand.ModuleJSON, string) {
	// This can't find packages in the stdlib, but that's harmless for all
	// the existing code paths.
	for _, m := range r.modsByModPath {
		if !strings.HasPrefix(importPath, m.Path) {
			continue
		}
		pathInModule := importPath[len(m.Path):]
		pkgDir := filepath.Join(m.Dir, pathInModule)
		if r.dirIsNestedModule(pkgDir, m) {
			continue
		}

		if info, ok := r.cacheLoad(pkgDir); ok {
			if loaded, err := info.reachedStatus(nameLoaded); loaded {
				if err != nil {
					continue // No package in this dir.
				}
				return m, pkgDir
			}
			if scanned, err := info.reachedStatus(directoryScanned); scanned && err != nil {
				continue // Dir is unreadable, etc.
			}
			// This is slightly wrong: a directory doesn't have to have an
			// importable package to count as a package for package-to-module
			// resolution. package main or _test files should count but
			// don't.
			// TODO(heschi): fix this.
			if _, err := r.cachePackageName(info); err == nil {
				return m, pkgDir
			}
		}

		// Not cached. Read the filesystem.
		pkgFiles, err := ioutil.ReadDir(pkgDir)
		if err != nil {
			continue
		}
		// A module only contains a package if it has buildable go
		// files in that directory. If not, it could be provided by an
		// outer module. See #29736.
		for _, fi := range pkgFiles {
			if ok, _ := r.env.matchFile(pkgDir, fi.Name()); ok {
				return m, pkgDir
			}
		}
	}
	return nil, ""
}

func (r *ModuleResolver) cacheLoad(dir string) (directoryPackageInfo, bool) {
	if info, ok := r.moduleCacheCache.Load(dir); ok {
		return info, ok
	}
	return r.otherCache.Load(dir)
}

func (r *ModuleResolver) cacheStore(info directoryPackageInfo) {
	if info.rootType == gopathwalk.RootModuleCache {
		r.moduleCacheCache.Store(info.dir, info)
	} else {
		r.otherCache.Store(info.dir, info)
	}
}

func (r *ModuleResolver) cacheKeys() []string {
	return append(r.moduleCacheCache.Keys(), r.otherCache.Keys()...)
}

// cachePackageName caches the package name for a dir already in the cache.
func (r *ModuleResolver) cachePackageName(info directoryPackageInfo) (string, error) {
	if info.rootType == gopathwalk.RootModuleCache {
		return r.moduleCacheCache.CachePackageName(info)
	}
	return r.otherCache.CachePackageName(info)
}

func (r *ModuleResolver) cacheExports(ctx context.Context, env *ProcessEnv, info directoryPackageInfo) (string, []string, error) {
	if info.rootType == gopathwalk.RootModuleCache {
		return r.moduleCacheCache.CacheExports(ctx, env, info)
	}
	return r.otherCache.CacheExports(ctx, env, info)
}

// findModuleByDir returns the module that contains dir, or nil if no such
// module is in scope.
func (r *ModuleResolver) findModuleByDir(dir string) *gocommand.ModuleJSON {
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

		if r.dirIsNestedModule(dir, m) {
			continue
		}

		return m
	}
	return nil
}

// dirIsNestedModule reports if dir is contained in a nested module underneath
// mod, not actually in mod.
func (r *ModuleResolver) dirIsNestedModule(dir string, mod *gocommand.ModuleJSON) bool {
	if !strings.HasPrefix(dir, mod.Dir) {
		return false
	}
	if r.dirInModuleCache(dir) {
		// Nested modules in the module cache are pruned,
		// so it cannot be a nested module.
		return false
	}
	if mod != nil && mod == r.dummyVendorMod {
		// The /vendor pseudomodule is flattened and doesn't actually count.
		return false
	}
	modDir, _ := r.modInfo(dir)
	if modDir == "" {
		return false
	}
	return modDir != mod.Dir
}

func (r *ModuleResolver) modInfo(dir string) (modDir string, modName string) {
	readModName := func(modFile string) string {
		modBytes, err := ioutil.ReadFile(modFile)
		if err != nil {
			return ""
		}
		return modulePath(modBytes)
	}

	if r.dirInModuleCache(dir) {
		if matches := modCacheRegexp.FindStringSubmatch(dir); len(matches) == 3 {
			index := strings.Index(dir, matches[1]+"@"+matches[2])
			modDir := filepath.Join(dir[:index], matches[1]+"@"+matches[2])
			return modDir, readModName(filepath.Join(modDir, "go.mod"))
		}
	}
	for {
		if info, ok := r.cacheLoad(dir); ok {
			return info.moduleDir, info.moduleName
		}
		f := filepath.Join(dir, "go.mod")
		info, err := os.Stat(f)
		if err == nil && !info.IsDir() {
			return dir, readModName(f)
		}

		d := filepath.Dir(dir)
		if len(d) >= len(dir) {
			return "", "" // reached top of file system, no go.mod
		}
		dir = d
	}
}

func (r *ModuleResolver) dirInModuleCache(dir string) bool {
	if r.moduleCacheDir == "" {
		return false
	}
	return strings.HasPrefix(dir, r.moduleCacheDir)
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

func (r *ModuleResolver) scan(ctx context.Context, callback *scanCallback) error {
	ctx, done := event.Start(ctx, "imports.ModuleResolver.scan")
	defer done()

	if err := r.init(); err != nil {
		return err
	}

	processDir := func(info directoryPackageInfo) {
		// Skip this directory if we were not able to get the package information successfully.
		if scanned, err := info.reachedStatus(directoryScanned); !scanned || err != nil {
			return
		}
		pkg, err := r.canonicalize(info)
		if err != nil {
			return
		}

		if !callback.dirFound(pkg) {
			return
		}
		pkg.packageName, err = r.cachePackageName(info)
		if err != nil {
			return
		}

		if !callback.packageNameLoaded(pkg) {
			return
		}
		_, exports, err := r.loadExports(ctx, pkg, false)
		if err != nil {
			return
		}
		callback.exportsLoaded(pkg, exports)
	}

	// Start processing everything in the cache, and listen for the new stuff
	// we discover in the walk below.
	stop1 := r.moduleCacheCache.ScanAndListen(ctx, processDir)
	defer stop1()
	stop2 := r.otherCache.ScanAndListen(ctx, processDir)
	defer stop2()

	// We assume cached directories are fully cached, including all their
	// children, and have not changed. We can skip them.
	skip := func(root gopathwalk.Root, dir string) bool {
		if r.env.SkipPathInScan != nil && root.Type == gopathwalk.RootCurrentModule {
			if root.Path == dir {
				return false
			}

			if r.env.SkipPathInScan(filepath.Clean(dir)) {
				return true
			}
		}

		info, ok := r.cacheLoad(dir)
		if !ok {
			return false
		}
		// This directory can be skipped as long as we have already scanned it.
		// Packages with errors will continue to have errors, so there is no need
		// to rescan them.
		packageScanned, _ := info.reachedStatus(directoryScanned)
		return packageScanned
	}

	// Add anything new to the cache, and process it if we're still listening.
	add := func(root gopathwalk.Root, dir string) {
		r.cacheStore(r.scanDirForPackage(root, dir))
	}

	// r.roots and the callback are not necessarily safe to use in the
	// goroutine below. Process them eagerly.
	roots := filterRoots(r.roots, callback.rootFound)
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
		// We have the lock on r.scannedRoots, and no other scans can run.
		for _, root := range roots {
			if ctx.Err() != nil {
				return
			}

			if r.scannedRoots[root] {
				continue
			}
			gopathwalk.WalkSkip([]gopathwalk.Root{root}, add, skip, gopathwalk.Options{Logf: r.env.Logf, ModulesEnabled: true})
			r.scannedRoots[root] = true
		}
		close(scanDone)
	}()
	select {
	case <-ctx.Done():
	case <-scanDone:
	}
	return nil
}

func (r *ModuleResolver) scoreImportPath(ctx context.Context, path string) float64 {
	if _, ok := stdlib[path]; ok {
		return MaxRelevance
	}
	mod, _ := r.findPackage(path)
	return modRelevance(mod)
}

func modRelevance(mod *gocommand.ModuleJSON) float64 {
	var relevance float64
	switch {
	case mod == nil: // out of scope
		return MaxRelevance - 4
	case mod.Indirect:
		relevance = MaxRelevance - 3
	case !mod.Main:
		relevance = MaxRelevance - 2
	default:
		relevance = MaxRelevance - 1 // main module ties with stdlib
	}

	_, versionString, ok := module.SplitPathVersion(mod.Path)
	if ok {
		index := strings.Index(versionString, "v")
		if index == -1 {
			return relevance
		}
		if versionNumber, err := strconv.ParseFloat(versionString[index+1:], 64); err == nil {
			relevance += versionNumber / 1000
		}
	}

	return relevance
}

// canonicalize gets the result of canonicalizing the packages using the results
// of initializing the resolver from 'go list -m'.
func (r *ModuleResolver) canonicalize(info directoryPackageInfo) (*pkg, error) {
	// Packages in GOROOT are already canonical, regardless of the std/cmd modules.
	if info.rootType == gopathwalk.RootGOROOT {
		return &pkg{
			importPathShort: info.nonCanonicalImportPath,
			dir:             info.dir,
			packageName:     path.Base(info.nonCanonicalImportPath),
			relevance:       MaxRelevance,
		}, nil
	}

	importPath := info.nonCanonicalImportPath
	mod := r.findModuleByDir(info.dir)
	// Check if the directory is underneath a module that's in scope.
	if mod != nil {
		// It is. If dir is the target of a replace directive,
		// our guessed import path is wrong. Use the real one.
		if mod.Dir == info.dir {
			importPath = mod.Path
		} else {
			dirInMod := info.dir[len(mod.Dir)+len("/"):]
			importPath = path.Join(mod.Path, filepath.ToSlash(dirInMod))
		}
	} else if !strings.HasPrefix(importPath, info.moduleName) {
		// The module's name doesn't match the package's import path. It
		// probably needs a replace directive we don't have.
		return nil, fmt.Errorf("package in %q is not valid without a replace statement", info.dir)
	}

	res := &pkg{
		importPathShort: importPath,
		dir:             info.dir,
		relevance:       modRelevance(mod),
	}
	// We may have discovered a package that has a different version
	// in scope already. Canonicalize to that one if possible.
	if _, canonicalDir := r.findPackage(importPath); canonicalDir != "" {
		res.dir = canonicalDir
	}
	return res, nil
}

func (r *ModuleResolver) loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []string, error) {
	if err := r.init(); err != nil {
		return "", nil, err
	}
	if info, ok := r.cacheLoad(pkg.dir); ok && !includeTest {
		return r.cacheExports(ctx, r.env, info)
	}
	return loadExportsFromFiles(ctx, r.env, pkg.dir, includeTest)
}

func (r *ModuleResolver) scanDirForPackage(root gopathwalk.Root, dir string) directoryPackageInfo {
	subdir := ""
	if dir != root.Path {
		subdir = dir[len(root.Path)+len("/"):]
	}
	importPath := filepath.ToSlash(subdir)
	if strings.HasPrefix(importPath, "vendor/") {
		// Only enter vendor directories if they're explicitly requested as a root.
		return directoryPackageInfo{
			status: directoryScanned,
			err:    fmt.Errorf("unwanted vendor directory"),
		}
	}
	switch root.Type {
	case gopathwalk.RootCurrentModule:
		importPath = path.Join(r.mainByDir[root.Path].Path, filepath.ToSlash(subdir))
	case gopathwalk.RootModuleCache:
		matches := modCacheRegexp.FindStringSubmatch(subdir)
		if len(matches) == 0 {
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("invalid module cache path: %v", subdir),
			}
		}
		modPath, err := module.UnescapePath(filepath.ToSlash(matches[1]))
		if err != nil {
			if r.env.Logf != nil {
				r.env.Logf("decoding module cache path %q: %v", subdir, err)
			}
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("decoding module cache path %q: %v", subdir, err),
			}
		}
		importPath = path.Join(modPath, filepath.ToSlash(matches[3]))
	}

	modDir, modName := r.modInfo(dir)
	result := directoryPackageInfo{
		status:                 directoryScanned,
		dir:                    dir,
		rootType:               root.Type,
		nonCanonicalImportPath: importPath,
		moduleDir:              modDir,
		moduleName:             modName,
	}
	if root.Type == gopathwalk.RootGOROOT {
		// stdlib packages are always in scope, despite the confusing go.mod
		return result
	}
	return result
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
