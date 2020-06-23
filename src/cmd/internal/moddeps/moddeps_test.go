// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package moddeps_test

import (
	"encoding/json"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"golang.org/x/mod/module"
)

type gorootModule struct {
	Path      string
	Dir       string
	hasVendor bool
}

// findGorootModules returns the list of modules found in the GOROOT source tree.
func findGorootModules(t *testing.T) []gorootModule {
	t.Helper()
	goBin := testenv.GoToolPath(t)

	goroot.once.Do(func() {
		goroot.err = filepath.Walk(runtime.GOROOT(), func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() && (info.Name() == "vendor" || info.Name() == "testdata") {
				return filepath.SkipDir
			}
			if path == filepath.Join(runtime.GOROOT(), "pkg") {
				// GOROOT/pkg contains generated artifacts, not source code.
				//
				// In https://golang.org/issue/37929 it was observed to somehow contain
				// a module cache, so it is important to skip. (That helps with the
				// running time of this test anyway.)
				return filepath.SkipDir
			}
			if info.IsDir() || info.Name() != "go.mod" {
				return nil
			}
			dir := filepath.Dir(path)

			// Use 'go list' to describe the module contained in this directory (but
			// not its dependencies).
			cmd := exec.Command(goBin, "list", "-json", "-m")
			cmd.Env = append(os.Environ(), "GO111MODULE=on")
			cmd.Dir = dir
			cmd.Stderr = new(strings.Builder)
			out, err := cmd.Output()
			if err != nil {
				return fmt.Errorf("'go list -json -m' in %s: %w\n%s", dir, err, cmd.Stderr)
			}

			var m gorootModule
			if err := json.Unmarshal(out, &m); err != nil {
				return fmt.Errorf("decoding 'go list -json -m' in %s: %w", dir, err)
			}
			if m.Path == "" || m.Dir == "" {
				return fmt.Errorf("'go list -json -m' in %s failed to populate Path and/or Dir", dir)
			}
			if _, err := os.Stat(filepath.Join(dir, "vendor")); err == nil {
				m.hasVendor = true
			}
			goroot.modules = append(goroot.modules, m)
			return nil
		})
	})

	if goroot.err != nil {
		t.Fatal(goroot.err)
	}
	return goroot.modules
}

// goroot caches the list of modules found in the GOROOT source tree.
var goroot struct {
	once    sync.Once
	modules []gorootModule
	err     error
}

// TestAllDependenciesVendored ensures that all packages imported within GOROOT
// are vendored in the corresponding GOROOT module.
//
// This property allows offline development within the Go project, and ensures
// that all dependency changes are presented in the usual code review process.
//
// This test does NOT ensure that the vendored contents match the unmodified
// contents of the corresponding dependency versions. Such as test would require
// network access, and would currently either need to copy the entire GOROOT module
// or explicitly invoke version control to check for changes.
// (See golang.org/issue/36852 and golang.org/issue/27348.)
func TestAllDependenciesVendored(t *testing.T) {
	goBin := testenv.GoToolPath(t)

	for _, m := range findGorootModules(t) {
		t.Run(m.Path, func(t *testing.T) {
			if m.hasVendor {
				// Load all of the packages in the module to ensure that their
				// dependencies are vendored. If any imported package is missing,
				// 'go list -deps' will fail when attempting to load it.
				cmd := exec.Command(goBin, "list", "-mod=vendor", "-deps", "./...")
				cmd.Env = append(os.Environ(), "GO111MODULE=on")
				cmd.Dir = m.Dir
				cmd.Stderr = new(strings.Builder)
				_, err := cmd.Output()
				if err != nil {
					t.Errorf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
					t.Logf("(Run 'go mod vendor' in %s to ensure that dependecies have been vendored.)", m.Dir)
				}
				return
			}

			// There is no vendor directory, so the module must have no dependencies.
			// Check that the list of active modules contains only the main module.
			cmd := exec.Command(goBin, "list", "-mod=mod", "-m", "all")
			cmd.Env = append(os.Environ(), "GO111MODULE=on")
			cmd.Dir = m.Dir
			cmd.Stderr = new(strings.Builder)
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, cmd.Stderr)
			}
			if strings.TrimSpace(string(out)) != m.Path {
				t.Errorf("'%s' reported active modules other than %s:\n%s", strings.Join(cmd.Args, " "), m.Path, out)
				t.Logf("(Run 'go mod tidy' in %s to ensure that no extraneous dependencies were added, or 'go mod vendor' to copy in imported packages.)", m.Dir)
			}
		})
	}
}

// TestDependencyVersionsConsistent verifies that each module in GOROOT that
// requires a given external dependency requires the same version of that
// dependency.
//
// This property allows us to maintain a single release branch of each such
// dependency, minimizing the number of backports needed to pull in critical
// fixes. It also ensures that any bug detected and fixed in one GOROOT module
// (such as "std") is fixed in all other modules (such as "cmd") as well.
func TestDependencyVersionsConsistent(t *testing.T) {
	// Collect the dependencies of all modules in GOROOT, indexed by module path.
	type requirement struct {
		Required    module.Version
		Replacement module.Version
	}
	seen := map[string]map[requirement][]gorootModule{} // module path → requirement → set of modules with that requirement
	for _, m := range findGorootModules(t) {
		if !m.hasVendor {
			// TestAllDependenciesVendored will ensure that the module has no
			// dependencies.
			continue
		}

		// We want this test to be able to run offline and with an empty module
		// cache, so we verify consistency only for the module versions listed in
		// vendor/modules.txt. That includes all direct dependencies and all modules
		// that provide any imported packages.
		//
		// It's ok if there are undetected differences in modules that do not
		// provide imported packages: we will not have to pull in any backports of
		// fixes to those modules anyway.
		vendor, err := ioutil.ReadFile(filepath.Join(m.Dir, "vendor", "modules.txt"))
		if err != nil {
			t.Error(err)
			continue
		}

		for _, line := range strings.Split(strings.TrimSpace(string(vendor)), "\n") {
			parts := strings.Fields(line)
			if len(parts) < 3 || parts[0] != "#" {
				continue
			}

			// This line is of the form "# module version [=> replacement [version]]".
			var r requirement
			r.Required.Path = parts[1]
			r.Required.Version = parts[2]
			if len(parts) >= 5 && parts[3] == "=>" {
				r.Replacement.Path = parts[4]
				if module.CheckPath(r.Replacement.Path) != nil {
					// If the replacement is a filesystem path (rather than a module path),
					// we don't know whether the filesystem contents have changed since
					// the module was last vendored.
					//
					// Fortunately, we do not currently use filesystem-local replacements
					// in GOROOT modules.
					t.Errorf("cannot check consistency for filesystem-local replacement in module %s (%s):\n%s", m.Path, m.Dir, line)
				}

				if len(parts) >= 6 {
					r.Replacement.Version = parts[5]
				}
			}

			if seen[r.Required.Path] == nil {
				seen[r.Required.Path] = make(map[requirement][]gorootModule)
			}
			seen[r.Required.Path][r] = append(seen[r.Required.Path][r], m)
		}
	}

	// Now verify that we saw only one distinct version for each module.
	for path, versions := range seen {
		if len(versions) > 1 {
			t.Errorf("Modules within GOROOT require different versions of %s.", path)
			for r, mods := range versions {
				desc := new(strings.Builder)
				desc.WriteString(r.Required.Version)
				if r.Replacement.Path != "" {
					fmt.Fprintf(desc, " => %s", r.Replacement.Path)
					if r.Replacement.Version != "" {
						fmt.Fprintf(desc, " %s", r.Replacement.Version)
					}
				}

				for _, m := range mods {
					t.Logf("%s\trequires %v", m.Path, desc)
				}
			}
		}
	}
}
