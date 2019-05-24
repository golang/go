// +build go1.11

package imports

import (
	"archive/zip"
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/internal/module"
	"golang.org/x/tools/internal/txtar"
)

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

	scan, err := mt.resolver.scan(nil)
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

// Tests that -mod=vendor sort of works. Adapted from mod_getmode_vendor.txt.
func TestModeGetmodeVendor(t *testing.T) {
	mt := setup(t, `
-- go.mod --
module x

require rsc.io/quote v1.5.2
-- x.go --
package x
import _ "rsc.io/quote"
`, "")
	defer mt.cleanup()

	if _, err := mt.env.invokeGo("mod", "vendor"); err != nil {
		t.Fatal(err)
	}

	mt.env.GOFLAGS = "-mod=vendor"
	mt.assertModuleFoundInDir("rsc.io/quote", "quote", `/vendor/`)

	mt.env.GOFLAGS = ""
	// Clear out the resolver's cache, since we've changed the environment.
	mt.resolver = &moduleResolver{env: mt.env}
	mt.assertModuleFoundInDir("rsc.io/quote", "quote", `pkg.*mod.*/quote@.*$`)
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
	scan, err := t.resolver.scan(nil)
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
	resolver *moduleResolver
	cleanup  func()
}

// setup builds a test enviroment from a txtar and supporting modules
// in testdata/mod, along the lines of TestScript in cmd/go.
func setup(t *testing.T, main, wd string) *modTest {
	t.Helper()
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
		GOPROXY:     proxyDirToURL(proxyDir),
		GOSUMDB:     "off",
		WorkingDir:  filepath.Join(mainDir, wd),
	}

	// go mod tidy instead of download because tidy will notice dependencies
	// in code, not just in go.mod files.
	if _, err := env.invokeGo("mod", "download"); err != nil {
		t.Fatal(err)
	}

	return &modTest{
		T:        t,
		env:      env,
		resolver: &moduleResolver{env: env},
		cleanup: func() {
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
		},
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
	modPath, err := module.DecodePath(modDir)
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
