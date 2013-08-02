package ssa_test

// This file runs the SSA builder in sanity-checking mode on all
// packages beneath $GOROOT and prints some summary information.
//
// Run test with GOMAXPROCS=8 and CGO_ENABLED=0.  The latter cannot be
// set from the test because it's too late to stop go/build.init()
// from picking up the value from the parent's environment.

import (
	"go/build"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
)

const debugMode = false

func allPackages() []string {
	var pkgs []string
	root := filepath.Join(runtime.GOROOT(), "src/pkg") + string(os.PathSeparator)
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		// Prune the search if we encounter any of these names:
		switch filepath.Base(path) {
		case "testdata", ".hg":
			return filepath.SkipDir
		}
		if info.IsDir() {
			pkg := strings.TrimPrefix(path, root)
			switch pkg {
			case "builtin", "pkg", "code.google.com":
				return filepath.SkipDir // skip these subtrees
			case "":
				return nil // ignore root of tree
			}
			pkgs = append(pkgs, pkg)
		}

		return nil
	})
	return pkgs
}

func TestStdlib(t *testing.T) {
	ctxt := build.Default
	ctxt.CgoEnabled = false
	impctx := importer.Config{Loader: importer.MakeGoBuildLoader(&ctxt)}

	// Load, parse and type-check the program.
	t0 := time.Now()

	var hasErrors bool
	imp := importer.New(&impctx)
	for _, importPath := range allPackages() {
		if _, err := imp.LoadPackage(importPath); err != nil {
			t.Errorf("LoadPackage(%s): %s", importPath, err)
			hasErrors = true
		}
	}

	t1 := time.Now()

	runtime.GC()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	alloc := memstats.Alloc

	// Create SSA packages.
	prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
	for _, info := range imp.Packages {
		if info.Err == nil {
			prog.CreatePackage(info).SetDebugMode(debugMode)
		}
	}

	t2 := time.Now()

	// Build SSA IR... if it's safe.
	if !hasErrors {
		prog.BuildAll()
	}

	t3 := time.Now()

	runtime.GC()
	runtime.ReadMemStats(&memstats)

	numPkgs := len(prog.PackagesByPath)
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Dump some statistics.
	allFuncs := ssa.AllFunctions(prog)
	var numInstrs int
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			numInstrs += len(b.Instrs)
		}
	}

	t.Log("GOMAXPROCS:           ", runtime.GOMAXPROCS(0))
	t.Log("Load/parse/typecheck: ", t1.Sub(t0))
	t.Log("SSA create:           ", t2.Sub(t1))
	if !hasErrors {
		t.Log("SSA build:            ", t3.Sub(t2))
	}

	// SSA stats:
	t.Log("#Packages:            ", numPkgs)
	t.Log("#Functions:           ", len(allFuncs))
	t.Log("#Instructions:        ", numInstrs)
	t.Log("#MB:                  ", (memstats.Alloc-alloc)/1000000)
}
