// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer_test

import (
	"fmt"
	"go/build"
	"testing"

	"code.google.com/p/go.tools/importer"
)

func TestLoadInitialPackages(t *testing.T) {
	ctxt := &importer.Config{Build: &build.Default}

	// Failed load: bad first import path causes parsePackageFiles to fail.
	args := []string{"nosuchpkg", "errors"}
	if _, _, err := importer.New(ctxt).LoadInitialPackages(args); err == nil {
		t.Errorf("LoadInitialPackages(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok.
	}

	// Failed load: bad second import path proceeds to doImport0, which fails.
	args = []string{"errors", "nosuchpkg"}
	if _, _, err := importer.New(ctxt).LoadInitialPackages(args); err == nil {
		t.Errorf("LoadInitialPackages(%q) succeeded, want failure", args)
	} else {
		// cannot find package: ok
	}

	// Successful load.
	args = []string{"fmt", "errors", "testdata/a.go,testdata/b.go", "--", "surplus"}
	imp := importer.New(ctxt)
	infos, rest, err := imp.LoadInitialPackages(args)
	if err != nil {
		t.Errorf("LoadInitialPackages(%q) failed: %s", args, err)
		return
	}
	if got, want := fmt.Sprint(rest), "[surplus]"; got != want {
		t.Errorf("LoadInitialPackages(%q) rest: got %s, want %s", got, want)
	}
	// Check list of initial packages.
	var pkgnames []string
	for _, info := range infos {
		pkgnames = append(pkgnames, info.Pkg.Path())
	}
	// Only the first import path (currently) contributes tests.
	if got, want := fmt.Sprint(pkgnames), "[fmt fmt_test errors P]"; got != want {
		t.Errorf("InitialPackages: got %s, want %s", got, want)
	}
	// Check set of transitive packages.
	// There are >30 and the set may grow over time, so only check a few.
	all := map[string]struct{}{}
	for _, info := range imp.AllPackages() {
		all[info.Pkg.Path()] = struct{}{}
	}
	want := []string{"strings", "time", "runtime", "testing", "unicode"}
	for _, w := range want {
		if _, ok := all[w]; !ok {
			t.Errorf("AllPackages: want element %s, got set %v", w, all)
		}
	}
}
