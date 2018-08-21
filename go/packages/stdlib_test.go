// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages_test

import (
	"bytes"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/go/packages"
)

// This test loads the metadata for the standard library,
func TestStdlibMetadata(t *testing.T) {
	// TODO(adonovan): see if we can get away without this hack.
	// if runtime.GOOS == "android" {
	// 	t.Skipf("incomplete std lib on %s", runtime.GOOS)
	// }

	runtime.GC()
	t0 := time.Now()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	alloc := memstats.Alloc

	// Load, parse and type-check the program.
	cfg := &packages.Config{Mode: packages.LoadAllSyntax}
	pkgs, err := packages.Load(cfg, "std")
	if err != nil {
		t.Fatalf("failed to load metadata: %v", err)
	}

	t1 := time.Now()
	runtime.GC()
	runtime.ReadMemStats(&memstats)
	runtime.KeepAlive(pkgs)

	t.Logf("Loaded %d packages", len(pkgs))
	numPkgs := len(pkgs)

	want := 150 // 186 on linux, 185 on windows.
	if numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	t.Log("GOMAXPROCS: ", runtime.GOMAXPROCS(0))
	t.Log("Metadata:   ", t1.Sub(t0))                          // ~800ms on 12 threads
	t.Log("#MB:        ", int64(memstats.Alloc-alloc)/1000000) // ~1MB
}

func TestCgoOption(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; uses tons of memory (golang.org/issue/14113)")
	}

	// TODO(adonovan): see if we can get away without these old
	// go/loader hacks now that we use the go list command.
	//
	// switch runtime.GOOS {
	// // On these systems, the net and os/user packages don't use cgo
	// // or the std library is incomplete (Android).
	// case "android", "plan9", "solaris", "windows":
	// 	t.Skipf("no cgo or incomplete std lib on %s", runtime.GOOS)
	// }
	// // In nocgo builds (e.g. linux-amd64-nocgo),
	// // there is no "runtime/cgo" package,
	// // so cgo-generated Go files will have a failing import.
	// if !build.Default.CgoEnabled {
	// 	return
	// }

	// Test that we can load cgo-using packages with
	// DisableCgo=true/false, which, among other things, causes go
	// list to select pure Go/native implementations, respectively,
	// based on build tags.
	//
	// Each entry specifies a package-level object and the generic
	// file expected to define it when cgo is disabled.
	// When cgo is enabled, the exact file is not specified (since
	// it varies by platform), but must differ from the generic one.
	//
	// The test also loads the actual file to verify that the
	// object is indeed defined at that location.
	for _, test := range []struct {
		pkg, name, genericFile string
	}{
		{"net", "cgoLookupHost", "cgo_stub.go"},
		{"os/user", "current", "lookup_stubs.go"},
	} {
		cfg := &packages.Config{Mode: packages.LoadSyntax}
		pkgs, err := packages.Load(cfg, test.pkg)
		if err != nil {
			t.Errorf("Load failed: %v", err)
			continue
		}
		pkg := pkgs[0]
		obj := pkg.Types.Scope().Lookup(test.name)
		if obj == nil {
			t.Errorf("no object %s.%s", test.pkg, test.name)
			continue
		}
		posn := pkg.Fset.Position(obj.Pos())
		gotFile := filepath.Base(posn.Filename)
		filesMatch := gotFile == test.genericFile

		if filesMatch {
			t.Errorf("!DisableCgo: %s found in %s, want native file",
				obj, gotFile)
		}

		// Load the file and check the object is declared at the right place.
		b, err := ioutil.ReadFile(posn.Filename)
		if err != nil {
			t.Errorf("can't read %s: %s", posn.Filename, err)
			continue
		}
		line := string(bytes.Split(b, []byte("\n"))[posn.Line-1])
		// Don't assume posn.Column is accurate.
		if !strings.Contains(line, "func "+test.name) {
			t.Errorf("%s: %s not declared here (looking at %q)", posn, obj, line)
		}
	}
}
