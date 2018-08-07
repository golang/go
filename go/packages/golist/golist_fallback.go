// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package golist

import (
	"context"
	"encoding/json"
	"fmt"

	"go/build"
	"golang.org/x/tools/go/internal/cgo"
	"golang.org/x/tools/go/packages/raw"
	"golang.org/x/tools/imports"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// TODO(matloob): Delete this file once Go 1.12 is released.

// This file provides backwards compatibility support for
// loading for versions of Go earlier than 1.10.4. This support is meant to
// assist with migration to the Package API until there's
// widespread adoption of these newer Go versions.
// This support will be removed once Go 1.12 is released
// in Q1 2019.

func golistPackagesFallback(ctx context.Context, cfg *raw.Config, words ...string) ([]string, []*raw.Package, error) {
	original, deps, err := getDeps(ctx, cfg, words...)
	if err != nil {
		return nil, nil, err
	}

	var tmpdir string // used for generated cgo files

	var result []*raw.Package
	var roots []string
	addPackage := func(p *jsonPackage) {
		if p.Name == "" {
			return
		}

		id := p.ImportPath
		isRoot := original[id] != nil
		pkgpath := id

		if pkgpath == "unsafe" {
			p.GoFiles = nil // ignore fake unsafe.go file
		}

		importMap := func(importlist []string) map[string]string {
			importMap := make(map[string]string)
			for _, id := range importlist {

				if id == "C" {
					importMap["unsafe"] = "unsafe"
					importMap["syscall"] = "syscall"
					if pkgpath != "runtime/cgo" {
						importMap["runtime/cgo"] = "runtime/cgo"
					}
					continue
				}
				importMap[imports.VendorlessPath(id)] = id
			}
			return importMap
		}
		if isRoot {
			roots = append(roots, id)
		}
		compiledGoFiles := absJoin(p.Dir, p.GoFiles)
		// Use a function to simplify control flow. It's just a bunch of gotos.
		processCgo := func() bool {
			// Suppress any cgo errors. Any relevant errors will show up in typechecking.
			// TODO(matloob): Skip running cgo if Mode < LoadTypes.
			if tmpdir == "" {
				if tmpdir, err = ioutil.TempDir("", "gopackages"); err != nil {
					return false
				}
			}
			outdir := filepath.Join(tmpdir, strings.Replace(p.ImportPath, "/", "_", -1))
			if err := os.Mkdir(outdir, 0755); err != nil {
				return false
			}
			files, _, err := runCgo(p.Dir, outdir, cfg.Env)
			if err != nil {
				return false
			}
			compiledGoFiles = append(compiledGoFiles, files...)
			return true
		}
		if len(p.CgoFiles) == 0 || !processCgo() {
			compiledGoFiles = append(compiledGoFiles, absJoin(p.Dir, p.CgoFiles)...) // Punt to typechecker.
		}
		result = append(result, &raw.Package{
			ID:              id,
			Name:            p.Name,
			GoFiles:         absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			CompiledGoFiles: compiledGoFiles, // TODO(matloob): Use cgo-processed Go files instead of p.GoFiles
			OtherFiles:      absJoin(p.Dir, p.SFiles, p.CFiles),
			PkgPath:         pkgpath,
			Imports:         importMap(p.Imports),
		})
		if cfg.Tests {
			testID := fmt.Sprintf("%s [%s.test]", id, id)
			if len(p.TestGoFiles) > 0 || len(p.XTestGoFiles) > 0 {
				if isRoot {
					roots = append(roots, testID)
				}
				result = append(result, &raw.Package{
					ID:              testID,
					Name:            p.Name,
					GoFiles:         absJoin(p.Dir, p.GoFiles, p.CgoFiles, p.TestGoFiles),
					CompiledGoFiles: append(compiledGoFiles, absJoin(p.Dir, p.TestGoFiles)...), // TODO(matloob): Can there be cgo files in the tests?
					OtherFiles:      absJoin(p.Dir, p.SFiles, p.CFiles),
					PkgPath:         pkgpath,
					Imports:         importMap(append(p.Imports, p.TestImports...)),
				})
			}
			if len(p.XTestGoFiles) > 0 {
				xtestID := fmt.Sprintf("%s_test [%s.test]", id, id)
				if isRoot {
					roots = append(roots, xtestID)
				}
				for i, imp := range p.XTestImports {
					if imp == p.ImportPath {
						p.XTestImports[i] = testID
						break
					}
				}
				result = append(result, &raw.Package{
					ID:              xtestID,
					Name:            p.Name + "_test",
					GoFiles:         absJoin(p.Dir, p.XTestGoFiles),
					CompiledGoFiles: absJoin(p.Dir, p.XTestGoFiles),
					PkgPath:         pkgpath,
					Imports:         importMap(p.XTestImports),
				})
			}
		}
	}

	for _, pkg := range original {
		addPackage(pkg)
	}
	if !cfg.Deps || len(deps) == 0 {
		return roots, result, nil
	}

	buf, err := golist(ctx, cfg, golistArgsFallback(cfg, deps))
	if err != nil {
		return nil, nil, err
	}

	// Decode the JSON and convert it to Package form.
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, nil, fmt.Errorf("JSON decoding failed: %v", err)
		}

		addPackage(p)
	}

	return roots, result, nil
}

// getDeps runs an initial go list to determine all the dependency packages.
func getDeps(ctx context.Context, cfg *raw.Config, words ...string) (originalSet map[string]*jsonPackage, deps []string, err error) {
	buf, err := golist(ctx, cfg, golistArgsFallback(cfg, words))
	if err != nil {
		return nil, nil, err
	}

	depsSet := make(map[string]bool)
	originalSet = make(map[string]*jsonPackage)
	var testImports []string

	// Extract deps from the JSON.
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, nil, fmt.Errorf("JSON decoding failed: %v", err)
		}

		originalSet[p.ImportPath] = p
		for _, dep := range p.Deps {
			depsSet[dep] = true
		}
		if cfg.Tests {
			// collect the additional imports of the test packages.
			pkgTestImports := append(p.TestImports, p.XTestImports...)
			for _, imp := range pkgTestImports {
				if depsSet[imp] {
					continue
				}
				depsSet[imp] = true
				testImports = append(testImports, imp)
			}
		}
	}
	// Get the deps of the packages imported by tests.
	if len(testImports) > 0 {
		buf, err = golist(ctx, cfg, golistArgsFallback(cfg, testImports))
		if err != nil {
			return nil, nil, err
		}
		// Extract deps from the JSON.
		for dec := json.NewDecoder(buf); dec.More(); {
			p := new(jsonPackage)
			if err := dec.Decode(p); err != nil {
				return nil, nil, fmt.Errorf("JSON decoding failed: %v", err)
			}
			for _, dep := range p.Deps {
				depsSet[dep] = true
			}
		}
	}

	for orig := range originalSet {
		delete(depsSet, orig)
	}

	deps = make([]string, 0, len(depsSet))
	for dep := range depsSet {
		deps = append(deps, dep)
	}
	return originalSet, deps, nil
}

func golistArgsFallback(cfg *raw.Config, words []string) []string {
	fullargs := []string{"list", "-e", "-json"}
	fullargs = append(fullargs, cfg.Flags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}

func runCgo(pkgdir, tmpdir string, env []string) (files, displayfiles []string, err error) {
	// Use go/build to open cgo files and determine the cgo flags, etc, from them.
	// This is tricky so it's best to avoid reimplementing as much as we can, and
	// we plan to delete this support once Go 1.12 is released anyways.
	// TODO(matloob): This isn't completely correct because we're using the Default
	// context. Perhaps we should more accurately fill in the context.
	bp, err := build.ImportDir(pkgdir, build.ImportMode(0))
	if err != nil {
		return nil, nil, err
	}
	for _, ev := range env {
		if v := strings.TrimPrefix(ev, "CGO_CPPFLAGS"); v != ev {
			bp.CgoCPPFLAGS = append(bp.CgoCPPFLAGS, strings.Fields(v)...)
		} else if v := strings.TrimPrefix(ev, "CGO_CFLAGS"); v != ev {
			bp.CgoCFLAGS = append(bp.CgoCFLAGS, strings.Fields(v)...)
		} else if v := strings.TrimPrefix(ev, "CGO_CXXFLAGS"); v != ev {
			bp.CgoCXXFLAGS = append(bp.CgoCXXFLAGS, strings.Fields(v)...)
		} else if v := strings.TrimPrefix(ev, "CGO_LDFLAGS"); v != ev {
			bp.CgoLDFLAGS = append(bp.CgoLDFLAGS, strings.Fields(v)...)
		}
	}
	return cgo.RunCgo(bp, pkgdir, tmpdir, true)
}
