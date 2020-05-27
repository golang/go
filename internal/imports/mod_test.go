package imports

import (
	"archive/zip"
	"context"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/proxydir"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// Tests that we can find packages in the stdlib.
func TestScanStdlib(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x
`, "")
	defer mt.cleanup()

	mt.assertScanFinds("fmt", "fmt")
}

// Tests that we handle a nested module. This is different from other tests
// where the module is in scope -- here we have to figure out the import path
// without any help from go list.
func TestScanOutOfScopeNestedModule(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

-- x.go --
package x

-- v2/go.mod --
module x

-- v2/x.go --
package x`, "")
	defer mt.cleanup()

	pkg := mt.assertScanFinds("x/v2", "x")
	if pkg != nil && !strings.HasSuffix(filepath.ToSlash(pkg.dir), "main/v2") {
		t.Errorf("x/v2 was found in %v, wanted .../main/v2", pkg.dir)
	}
	// We can't load the package name from the import path, but that should
	// be okay -- if we end up adding this result, we'll add it with a name
	// if necessary.
}

// Tests that we don't find a nested module contained in a local replace target.
// The code for this case is too annoying to write, so it's just ignored.
func TestScanNestedModuleInLocalReplace(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require y v0.0.0
replace y => ./y

-- x.go --
package x

-- y/go.mod --
module y

-- y/y.go --
package y

-- y/z/go.mod --
module y/z

-- y/z/z.go --
package z
`, "")
	defer mt.cleanup()

	mt.assertFound("y", "y")

	scan, err := scanToSlice(mt.resolver, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range scan {
		if strings.HasSuffix(filepath.ToSlash(pkg.dir), "main/y/z") {
			t.Errorf("scan found a package %v in dir main/y/z, wanted none", pkg.importPathShort)
		}
	}
}

// Tests that path encoding is handled correctly. Adapted from mod_case.txt.
func TestModCase(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require rsc.io/QUOTE v1.5.2

-- x.go --
package x

import _ "rsc.io/QUOTE/QUOTE"
`, "")
	defer mt.cleanup()
	mt.assertFound("rsc.io/QUOTE/QUOTE", "QUOTE")
}

// Not obviously relevant to goimports. Adapted from mod_domain_root.txt anyway.
func TestModDomainRoot(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require example.com v1.0.0

-- x.go --
package x
import _ "example.com"
`, "")
	defer mt.cleanup()
	mt.assertFound("example.com", "x")
}

// Tests that scanning the module cache > 1 time is able to find the same module.
func TestModMultipleScans(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require example.com v1.0.0

-- x.go --
package x
import _ "example.com"
`, "")
	defer mt.cleanup()

	mt.assertScanFinds("example.com", "x")
	mt.assertScanFinds("example.com", "x")
}

// Tests that scanning the module cache > 1 time is able to find the same module
// in the module cache.
func TestModMultipleScansWithSubdirs(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require rsc.io/quote v1.5.2

-- x.go --
package x
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()

	mt.assertScanFinds("rsc.io/quote", "quote")
	mt.assertScanFinds("rsc.io/quote", "quote")
}

// Tests that scanning the module cache > 1 after changing a package in module cache to make it unimportable
// is able to find the same module.
func TestModCacheEditModFile(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require rsc.io/quote v1.5.2
-- x.go --
package x
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()
	found := mt.assertScanFinds("rsc.io/quote", "quote")
	if found == nil {
		t.Fatal("rsc.io/quote not found in initial scan.")
	}

	// Update the go.mod file of example.com so that it changes its module path (not allowed).
	if err := os.Chmod(filepath.Join(found.dir, "go.mod"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(found.dir, "go.mod"), []byte("module bad.com\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Test that with its cache of module packages it still finds the package.
	mt.assertScanFinds("rsc.io/quote", "quote")

	// Rewrite the main package so that rsc.io/quote is not in scope.
	if err := ioutil.WriteFile(filepath.Join(mt.env.WorkingDir, "go.mod"), []byte("module x\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(mt.env.WorkingDir, "x.go"), []byte("package x\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Uninitialize the go.mod dependent cached information and make sure it still finds the package.
	mt.resolver.ClearForNewMod()
	mt.assertScanFinds("rsc.io/quote", "quote")
}

// Tests that -mod=vendor works. Adapted from mod_vendor_build.txt.
func TestModVendorBuild(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module m
go 1.12
require rsc.io/sampler v1.3.1
-- x.go --
package x
import _ "rsc.io/sampler"
`, "")
	defer mt.cleanup()

	// Sanity-check the setup.
	mt.assertModuleFoundInDir("rsc.io/sampler", "sampler", `pkg.*mod.*/sampler@.*$`)

	// Populate vendor/ and clear out the mod cache so we can't cheat.
	if _, err := mt.env.invokeGo(context.Background(), "mod", "vendor"); err != nil {
		t.Fatal(err)
	}
	if _, err := mt.env.invokeGo(context.Background(), "clean", "-modcache"); err != nil {
		t.Fatal(err)
	}

	// Clear out the resolver's cache, since we've changed the environment.
	mt.resolver = newModuleResolver(mt.env)
	mt.env.GOFLAGS = "-mod=vendor"
	mt.assertModuleFoundInDir("rsc.io/sampler", "sampler", `/vendor/`)
}

// Tests that -mod=vendor is auto-enabled only for go1.14 and higher.
// Vaguely inspired by mod_vendor_auto.txt.
func TestModVendorAuto(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module m
go 1.14
require rsc.io/sampler v1.3.1
-- x.go --
package x
import _ "rsc.io/sampler"
`, "")
	defer mt.cleanup()

	// Populate vendor/.
	if _, err := mt.env.invokeGo(context.Background(), "mod", "vendor"); err != nil {
		t.Fatal(err)
	}

	wantDir := `pkg.*mod.*/sampler@.*$`
	if testenv.Go1Point() >= 14 {
		wantDir = `/vendor/`
	}
	mt.assertModuleFoundInDir("rsc.io/sampler", "sampler", wantDir)
}

// Tests that a module replace works. Adapted from mod_list.txt. We start with
// go.mod2; the first part of the test is irrelevant.
func TestModList(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x
require rsc.io/quote v1.5.1
replace rsc.io/sampler v1.3.0 => rsc.io/sampler v1.3.1

-- x.go --
package x
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()

	mt.assertModuleFoundInDir("rsc.io/sampler", "sampler", `pkg.mod.*/sampler@v1.3.1$`)
}

// Tests that a local replace works. Adapted from mod_local_replace.txt.
func TestModLocalReplace(t *testing.T) {
	mt := setup(t, `
-- x/y/go.mod --
module x/y
require zz v1.0.0
replace zz v1.0.0 => ../z

-- x/y/y.go --
package y
import _ "zz"

-- x/z/go.mod --
module x/z

-- x/z/z.go --
package z
`, "x/y")
	defer mt.cleanup()

	mt.assertFound("zz", "z")
}

// Tests that the package at the root of the main module can be found.
// Adapted from the first part of mod_multirepo.txt.
func TestModMultirepo1(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module rsc.io/quote

-- x.go --
package quote
`, "")
	defer mt.cleanup()

	mt.assertModuleFoundInDir("rsc.io/quote", "quote", `/main`)
}

// Tests that a simple module dependency is found. Adapted from the third part
// of mod_multirepo.txt (We skip the case where it doesn't have a go.mod
// entry -- we just don't work in that case.)
func TestModMultirepo3(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module rsc.io/quote

require rsc.io/quote/v2 v2.0.1
-- x.go --
package quote

import _ "rsc.io/quote/v2"
`, "")
	defer mt.cleanup()

	mt.assertModuleFoundInDir("rsc.io/quote", "quote", `/main`)
	mt.assertModuleFoundInDir("rsc.io/quote/v2", "quote", `pkg.mod.*/v2@v2.0.1$`)
}

// Tests that a nested module is found in the module cache, even though
// it's checked out. Adapted from the fourth part of mod_multirepo.txt.
func TestModMultirepo4(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module rsc.io/quote
require rsc.io/quote/v2 v2.0.1

-- x.go --
package quote
import _ "rsc.io/quote/v2"

-- v2/go.mod --
package rsc.io/quote/v2

-- v2/x.go --
package quote
import _ "rsc.io/quote/v2"
`, "")
	defer mt.cleanup()

	mt.assertModuleFoundInDir("rsc.io/quote", "quote", `/main`)
	mt.assertModuleFoundInDir("rsc.io/quote/v2", "quote", `pkg.mod.*/v2@v2.0.1$`)
}

// Tests a simple module dependency. Adapted from the first part of mod_replace.txt.
func TestModReplace1(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module quoter

require rsc.io/quote/v3 v3.0.0

-- main.go --

package main
`, "")
	defer mt.cleanup()
	mt.assertFound("rsc.io/quote/v3", "quote")
}

// Tests a local replace. Adapted from the second part of mod_replace.txt.
func TestModReplace2(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module quoter

require rsc.io/quote/v3 v3.0.0
replace rsc.io/quote/v3 => ./local/rsc.io/quote/v3
-- main.go --
package main

-- local/rsc.io/quote/v3/go.mod --
module rsc.io/quote/v3

require rsc.io/sampler v1.3.0

-- local/rsc.io/quote/v3/quote.go --
package quote

import "rsc.io/sampler"
`, "")
	defer mt.cleanup()
	mt.assertModuleFoundInDir("rsc.io/quote/v3", "quote", `/local/rsc.io/quote/v3`)
}

// Tests that a module can be replaced by a different module path. Adapted
// from the third part of mod_replace.txt.
func TestModReplace3(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module quoter

require not-rsc.io/quote/v3 v3.1.0
replace not-rsc.io/quote/v3 v3.1.0 => ./local/rsc.io/quote/v3

-- usenewmodule/main.go --
package main

-- local/rsc.io/quote/v3/go.mod --
module rsc.io/quote/v3

require rsc.io/sampler v1.3.0

-- local/rsc.io/quote/v3/quote.go --
package quote

-- local/not-rsc.io/quote/v3/go.mod --
module not-rsc.io/quote/v3

-- local/not-rsc.io/quote/v3/quote.go --
package quote
`, "")
	defer mt.cleanup()
	mt.assertModuleFoundInDir("not-rsc.io/quote/v3", "quote", "local/rsc.io/quote/v3")
}

// Tests more local replaces, notably the case where an outer module provides
// a package that could also be provided by an inner module. Adapted from
// mod_replace_import.txt, with example.com/v changed to /vv because Go 1.11
// thinks /v is an invalid major version.
func TestModReplaceImport(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module example.com/m

replace (
	example.com/a => ./a
	example.com/a/b => ./b
)

replace (
	example.com/x => ./x
	example.com/x/v3 => ./v3
)

replace (
	example.com/y/z/w => ./w
	example.com/y => ./y
)

replace (
	example.com/vv v1.11.0 => ./v11
	example.com/vv v1.12.0 => ./v12
	example.com/vv => ./vv
)

require (
	example.com/a/b v0.0.0
	example.com/x/v3 v3.0.0
	example.com/y v0.0.0
	example.com/y/z/w v0.0.0
	example.com/vv v1.12.0
)
-- m.go --
package main
import (
	_ "example.com/a/b"
	_ "example.com/x/v3"
	_ "example.com/y/z/w"
	_ "example.com/vv"
)
func main() {}

-- a/go.mod --
module a.localhost
-- a/a.go --
package a
-- a/b/b.go--
package b

-- b/go.mod --
module a.localhost/b
-- b/b.go --
package b

-- x/go.mod --
module x.localhost
-- x/x.go --
package x
-- x/v3.go --
package v3
import _ "x.localhost/v3"

-- v3/go.mod --
module x.localhost/v3
-- v3/x.go --
package x

-- w/go.mod --
module w.localhost
-- w/skip/skip.go --
// Package skip is nested below nonexistent package w.
package skip

-- y/go.mod --
module y.localhost
-- y/z/w/w.go --
package w

-- v12/go.mod --
module v.localhost
-- v12/v.go --
package v

-- v11/go.mod --
module v.localhost
-- v11/v.go --
package v

-- vv/go.mod --
module v.localhost
-- vv/v.go --
package v
`, "")
	defer mt.cleanup()

	mt.assertModuleFoundInDir("example.com/a/b", "b", `main/b$`)
	mt.assertModuleFoundInDir("example.com/x/v3", "x", `main/v3$`)
	mt.assertModuleFoundInDir("example.com/y/z/w", "w", `main/y/z/w$`)
	mt.assertModuleFoundInDir("example.com/vv", "v", `main/v12$`)
}

// Tests that we handle GO111MODULE=on with no go.mod file. See #30855.
func TestNoMainModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 12)
	mt := setup(t, `
-- x.go --
package x
`, "")
	defer mt.cleanup()
	if _, err := mt.env.invokeGo(context.Background(), "mod", "download", "rsc.io/quote@v1.5.1"); err != nil {
		t.Fatal(err)
	}

	mt.assertScanFinds("rsc.io/quote", "quote")
}

// assertFound asserts that the package at importPath is found to have pkgName,
// and that scanning for pkgName finds it at importPath.
func (t *modTest) assertFound(importPath, pkgName string) (string, *pkg) {
	t.Helper()

	names, err := t.resolver.loadPackageNames([]string{importPath}, t.env.WorkingDir)
	if err != nil {
		t.Errorf("loading package name for %v: %v", importPath, err)
	}
	if names[importPath] != pkgName {
		t.Errorf("package name for %v = %v, want %v", importPath, names[importPath], pkgName)
	}
	pkg := t.assertScanFinds(importPath, pkgName)

	_, foundDir := t.resolver.findPackage(importPath)
	return foundDir, pkg
}

func (t *modTest) assertScanFinds(importPath, pkgName string) *pkg {
	t.Helper()
	scan, err := scanToSlice(t.resolver, nil)
	if err != nil {
		t.Errorf("scan failed: %v", err)
	}
	for _, pkg := range scan {
		if pkg.importPathShort == importPath {
			return pkg
		}
	}
	t.Errorf("scanning for %v did not find %v", pkgName, importPath)
	return nil
}

func scanToSlice(resolver Resolver, exclude []gopathwalk.RootType) ([]*pkg, error) {
	var mu sync.Mutex
	var result []*pkg
	filter := &scanCallback{
		rootFound: func(root gopathwalk.Root) bool {
			for _, rt := range exclude {
				if root.Type == rt {
					return false
				}
			}
			return true
		},
		dirFound: func(pkg *pkg) bool {
			return true
		},
		packageNameLoaded: func(pkg *pkg) bool {
			mu.Lock()
			defer mu.Unlock()
			result = append(result, pkg)
			return false
		},
	}
	err := resolver.scan(context.Background(), filter)
	return result, err
}

// assertModuleFoundInDir is the same as assertFound, but also checks that the
// package was found in an active module whose Dir matches dirRE.
func (t *modTest) assertModuleFoundInDir(importPath, pkgName, dirRE string) {
	t.Helper()
	dir, pkg := t.assertFound(importPath, pkgName)
	re, err := regexp.Compile(dirRE)
	if err != nil {
		t.Fatal(err)
	}

	if dir == "" {
		t.Errorf("import path %v not found in active modules", importPath)
	} else {
		if !re.MatchString(filepath.ToSlash(dir)) {
			t.Errorf("finding dir for %s: dir = %q did not match regex %q", importPath, dir, dirRE)
		}
	}
	if pkg != nil {
		if !re.MatchString(filepath.ToSlash(pkg.dir)) {
			t.Errorf("scanning for %s: dir = %q did not match regex %q", pkgName, pkg.dir, dirRE)
		}
	}
}

var proxyOnce sync.Once
var proxyDir string

type modTest struct {
	*testing.T
	env      *ProcessEnv
	resolver *ModuleResolver
	cleanup  func()
}

// setup builds a test environment from a txtar and supporting modules
// in testdata/mod, along the lines of TestScript in cmd/go.
func setup(t *testing.T, main, wd string) *modTest {
	t.Helper()
	testenv.NeedsGo1Point(t, 11)
	testenv.NeedsTool(t, "go")

	proxyOnce.Do(func() {
		var err error
		proxyDir, err = ioutil.TempDir("", "proxy-")
		if err != nil {
			t.Fatal(err)
		}
		if err := writeProxy(proxyDir, "testdata/mod"); err != nil {
			t.Fatal(err)
		}
	})

	dir, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}

	mainDir := filepath.Join(dir, "main")
	if err := writeModule(mainDir, main); err != nil {
		t.Fatal(err)
	}

	env := &ProcessEnv{
		GOROOT:      build.Default.GOROOT,
		GOPATH:      filepath.Join(dir, "gopath"),
		GO111MODULE: "on",
		GOPROXY:     proxydir.ToURL(proxyDir),
		GOSUMDB:     "off",
		WorkingDir:  filepath.Join(mainDir, wd),
		GocmdRunner: &gocommand.Runner{},
	}
	if *testDebug {
		env.Logf = log.Printf
	}

	// go mod download gets mad if we don't have a go.mod, so make sure we do.
	_, err = os.Stat(filepath.Join(mainDir, "go.mod"))
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("checking if go.mod exists: %v", err)
	}
	if err == nil {
		if _, err := env.invokeGo(context.Background(), "mod", "download"); err != nil {
			t.Fatal(err)
		}
	}

	return &modTest{
		T:        t,
		env:      env,
		resolver: newModuleResolver(env),
		cleanup:  func() { removeDir(dir) },
	}
}

// writeModule writes the module in the ar, a txtar, to dir.
func writeModule(dir, ar string) error {
	a := txtar.Parse([]byte(ar))

	for _, f := range a.Files {
		fpath := filepath.Join(dir, f.Name)
		if err := os.MkdirAll(filepath.Dir(fpath), 0755); err != nil {
			return err
		}

		if err := ioutil.WriteFile(fpath, f.Data, 0644); err != nil {
			return err
		}
	}
	return nil
}

// writeProxy writes all the txtar-formatted modules in arDir to a proxy
// directory in dir.
func writeProxy(dir, arDir string) error {
	files, err := ioutil.ReadDir(arDir)
	if err != nil {
		return err
	}

	for _, fi := range files {
		if err := writeProxyModule(dir, filepath.Join(arDir, fi.Name())); err != nil {
			return err
		}
	}
	return nil
}

// writeProxyModule writes a txtar-formatted module at arPath to the module
// proxy in base.
func writeProxyModule(base, arPath string) error {
	arName := filepath.Base(arPath)
	i := strings.LastIndex(arName, "_v")
	ver := strings.TrimSuffix(arName[i+1:], ".txt")
	modDir := strings.Replace(arName[:i], "_", "/", -1)
	modPath, err := module.UnescapePath(modDir)
	if err != nil {
		return err
	}

	dir := filepath.Join(base, modDir, "@v")
	a, err := txtar.ParseFile(arPath)

	if err != nil {
		return err
	}

	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	f, err := os.OpenFile(filepath.Join(dir, ver+".zip"), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	z := zip.NewWriter(f)
	for _, f := range a.Files {
		if f.Name[0] == '.' {
			if err := ioutil.WriteFile(filepath.Join(dir, ver+f.Name), f.Data, 0644); err != nil {
				return err
			}
		} else {
			zf, err := z.Create(modPath + "@" + ver + "/" + f.Name)
			if err != nil {
				return err
			}
			if _, err := zf.Write(f.Data); err != nil {
				return err
			}
		}
	}
	if err := z.Close(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	list, err := os.OpenFile(filepath.Join(dir, "list"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(list, "%s\n", ver); err != nil {
		return err
	}
	if err := list.Close(); err != nil {
		return err
	}
	return nil
}

func removeDir(dir string) {
	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			_ = os.Chmod(path, 0777)
		}
		return nil
	})
	_ = os.RemoveAll(dir) // ignore errors
}

// Tests that findModFile can find the mod files from a path in the module cache.
func TestFindModFileModCache(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require rsc.io/quote v1.5.2
-- x.go --
package x
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()
	want := filepath.Join(mt.resolver.env.GOPATH, "pkg/mod", "rsc.io/quote@v1.5.2")

	found := mt.assertScanFinds("rsc.io/quote", "quote")
	modDir, _ := mt.resolver.modInfo(found.dir)
	if modDir != want {
		t.Errorf("expected: %s, got: %s", want, modDir)
	}
}

// Tests that crud in the module cache is ignored.
func TestInvalidModCache(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	dir, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer removeDir(dir)

	// This doesn't have module@version like it should.
	if err := os.MkdirAll(filepath.Join(dir, "gopath/pkg/mod/sabotage"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "gopath/pkg/mod/sabotage/x.go"), []byte("package foo\n"), 0777); err != nil {
		t.Fatal(err)
	}
	env := &ProcessEnv{
		GOROOT:      build.Default.GOROOT,
		GOPATH:      filepath.Join(dir, "gopath"),
		GO111MODULE: "on",
		GOSUMDB:     "off",
		GocmdRunner: &gocommand.Runner{},
		WorkingDir:  dir,
	}
	resolver := newModuleResolver(env)
	scanToSlice(resolver, nil)
}

func TestGetCandidatesRanking(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module example.com

require rsc.io/quote v1.5.1

-- rpackage/x.go --
package rpackage
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()

	if _, err := mt.env.invokeGo(context.Background(), "mod", "download", "rsc.io/quote/v2@v2.0.1"); err != nil {
		t.Fatal(err)
	}

	type res struct {
		relevance  int
		name, path string
	}
	want := []res{
		// Stdlib
		{7, "bytes", "bytes"},
		{7, "http", "net/http"},
		// Main module
		{6, "rpackage", "example.com/rpackage"},
		// Direct module deps
		{5, "quote", "rsc.io/quote"},
		// Indirect deps
		{4, "language", "golang.org/x/text/language"},
		// Out of scope modules
		{3, "quote", "rsc.io/quote/v2"},
	}
	var mu sync.Mutex
	var got []res
	add := func(c ImportFix) {
		mu.Lock()
		defer mu.Unlock()
		for _, w := range want {
			if c.StmtInfo.ImportPath == w.path {
				got = append(got, res{c.Relevance, c.IdentName, c.StmtInfo.ImportPath})
			}
		}
	}
	if err := getAllCandidates(context.Background(), add, "", "foo.go", "foo", mt.env); err != nil {
		t.Fatalf("getAllCandidates() = %v", err)
	}
	sort.Slice(got, func(i, j int) bool {
		ri, rj := got[i], got[j]
		if ri.relevance != rj.relevance {
			return ri.relevance > rj.relevance // Highest first.
		}
		return ri.name < rj.name
	})
	if !reflect.DeepEqual(want, got) {
		t.Errorf("wanted candidates in order %v, got %v", want, got)
	}
}

func BenchmarkScanModCache(b *testing.B) {
	testenv.NeedsGo1Point(b, 11)
	env := &ProcessEnv{
		GOPATH:      build.Default.GOPATH,
		GOROOT:      build.Default.GOROOT,
		GocmdRunner: &gocommand.Runner{},
		Logf:        log.Printf,
	}
	exclude := []gopathwalk.RootType{gopathwalk.RootGOROOT}
	scanToSlice(env.GetResolver(), exclude)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scanToSlice(env.GetResolver(), exclude)
		env.GetResolver().(*ModuleResolver).ClearForNewScan()
	}
}
