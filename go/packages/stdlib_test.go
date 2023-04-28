// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages_test

import (
	"runtime"
	"testing"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/testenv"
)

// This test loads the metadata for the standard library,
func TestStdlibMetadata(t *testing.T) {
	testenv.NeedsGoPackages(t)

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
	if packages.PrintErrors(pkgs) > 0 {
		t.Fatal("there were errors loading standard library")
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
