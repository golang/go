// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testimporter

import (
	"bufio"
	"fmt"
	"go/build"
	"internal/exportdata"
	"internal/pkgbits"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"cmd/compile/internal/types2"
)

// Importer implements a types2 importer for use in testing by calling "go
// build". It is safe for concurrent use; sharing importers can yield better
// performance. It understands the compiler-internal unified export formats.
type Importer struct {
	dir      string                     // work directory
	mu       sync.Mutex                 // guards the fields below
	readPkgs map[string]*types2.Package // package path -> package
	bldOnces map[string]*sync.Once      // package path -> build function
	bldCache map[string]*bldResult      // package path -> build result
}

type bldResult struct {
	out string // path to built archive
	err error  // nil if compilation succeeded
}

// NewImporter returns a new Importer.
func NewImporter() *Importer {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		panic("could not create temp directory")
	}
	return &Importer{
		dir:      dir,
		mu:       sync.Mutex{},
		readPkgs: make(map[string]*types2.Package),
		bldOnces: make(map[string]*sync.Once),
		bldCache: make(map[string]*bldResult),
	}
}

// Import implements types2.Importer.
func (imp *Importer) Import(path string) (*types2.Package, error) {
	return imp.ImportFrom(path, "", 0)
}

// ImportFrom implements types2.ImportFrom.
func (imp *Importer) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	assert(mode == 0)
	if path == "unsafe" {
		return types2.Unsafe, nil
	}
	bld, err := build.Import(path, srcDir, build.FindOnly)
	if err != nil {
		return nil, err
	}
	// srcDir is only relevant if the package is not in GOROOT.
	if !bld.Goroot {
		assert(filepath.IsAbs(srcDir)) // see #14282
	}
	path = bld.ImportPath
	// If the package was already read (fully), avoid reading it again.
	// Note pkg.Complete must be observed with the lock since packages are modified concurrently.
	imp.mu.Lock()
	if pkg, ok := imp.readPkgs[path]; ok && pkg.Complete() {
		imp.mu.Unlock()
		return pkg, nil
	}
	imp.mu.Unlock()
	return imp.readArchive(path, bld.Dir)
}

func (imp *Importer) readArchive(path, dir string) (*types2.Package, error) {
	out, err := imp.compile(path, dir)
	if err != nil {
		return nil, err
	}
	// Open and decode the output.
	f, err := os.Open(out)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := bufio.NewReader(f)
	data, err := exportdata.ReadUnified(buf)
	if err != nil {
		return nil, err
	}
	// Guard writes to imp.readPkgs in ReadPackages.
	imp.mu.Lock()
	defer imp.mu.Unlock()
	// While ReadPackage might populate imp.readPkgs with an incomplete package,
	// we check for completeness before returning from ImportFrom.
	return ReadPackage(nil, imp.readPkgs, pkgbits.NewPkgDecoder(path, string(data))), nil
}

func (imp *Importer) compile(path, dir string) (string, error) {
	imp.mu.Lock()
	once, ok := imp.bldOnces[path]
	if !ok {
		once = &sync.Once{}
		imp.bldOnces[path] = once
	}
	imp.mu.Unlock()
	once.Do(func() {
		// We're first, do the build.
		out := filepath.Join(imp.dir, strings.ReplaceAll(path, "/", "_")+".a")
		cmd := exec.Command(filepath.Join(build.Default.GOROOT, "bin", "go"), "build", "-o", out, dir)
		var res *bldResult
		if bytes, err := cmd.CombinedOutput(); err != nil {
			res = &bldResult{err: fmt.Errorf("building %s failed: %s", path, bytes)}
		} else {
			res = &bldResult{out: out}
		}
		imp.mu.Lock()
		imp.bldCache[path] = res
		imp.mu.Unlock()
	})
	imp.mu.Lock()
	res := imp.bldCache[path]
	imp.mu.Unlock()
	return res.out, res.err
}
