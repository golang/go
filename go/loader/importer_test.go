// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader_test

import (
	"fmt"
	"sort"
	"testing"

	"code.google.com/p/go.tools/go/loader"
)

func loadFromArgs(args []string) (prog *loader.Program, rest []string, err error) {
	conf := &loader.Config{}
	rest, err = conf.FromArgs(args)
	if err == nil {
		prog, err = conf.Load()
	}
	return
}

func TestLoadFromArgs(t *testing.T) {
	// Failed load: bad first import path causes parsePackageFiles to fail.
	args := []string{"nosuchpkg", "errors"}
	if _, _, err := loadFromArgs(args); err == nil {
		t.Errorf("loadFromArgs(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok.
	}

	// Failed load: bad second import path proceeds to doImport0, which fails.
	args = []string{"errors", "nosuchpkg"}
	if _, _, err := loadFromArgs(args); err == nil {
		t.Errorf("loadFromArgs(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok
	}

	// Successful load.
	args = []string{"fmt", "errors", "testdata/a.go,testdata/b.go", "--", "surplus"}
	prog, rest, err := loadFromArgs(args)
	if err != nil {
		t.Errorf("loadFromArgs(%q) failed: %s", args, err)
		return
	}
	if got, want := fmt.Sprint(rest), "[surplus]"; got != want {
		t.Errorf("loadFromArgs(%q) rest: got %s, want %s", args, got, want)
	}
	// Check list of Created packages.
	var pkgnames []string
	for _, info := range prog.Created {
		pkgnames = append(pkgnames, info.Pkg.Path())
	}
	// Only the first import path (currently) contributes tests.
	if got, want := fmt.Sprint(pkgnames), "[fmt_test P]"; got != want {
		t.Errorf("Created: got %s, want %s", got, want)
	}

	// Check set of Imported packages.
	pkgnames = nil
	for path := range prog.Imported {
		pkgnames = append(pkgnames, path)
	}
	sort.Strings(pkgnames)
	// Only the first import path (currently) contributes tests.
	if got, want := fmt.Sprint(pkgnames), "[errors fmt]"; got != want {
		t.Errorf("Loaded: got %s, want %s", got, want)
	}

	// Check set of transitive packages.
	// There are >30 and the set may grow over time, so only check a few.
	all := map[string]struct{}{}
	for _, info := range prog.AllPackages {
		all[info.Pkg.Path()] = struct{}{}
	}
	want := []string{"strings", "time", "runtime", "testing", "unicode"}
	for _, w := range want {
		if _, ok := all[w]; !ok {
			t.Errorf("AllPackages: want element %s, got set %v", w, all)
		}
	}
}
