package packages

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/imports"
)

// TODO(matloob): Delete this file once Go 1.12 is released.

// This file provides backwards compatibility support for
// loading for versions of Go earlier than 1.10.4. This support is meant to
// assist with migration to the Package API until there's
// widespread adoption of these newer Go versions.
// This support will be removed once Go 1.12 is released
// in Q1 2019.

// TODO(matloob): Support cgo. Copy code from the loader that runs cgo.

func golistPackagesFallback(ctx context.Context, cfg *rawConfig, words ...string) ([]string, []*rawPackage, error) {
	original, deps, err := getDeps(ctx, cfg, words...)
	if err != nil {
		return nil, nil, err
	}

	var result []*rawPackage
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
		result = append(result, &rawPackage{
			ID:         id,
			Name:       p.Name,
			GoFiles:    absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			OtherFiles: absJoin(p.Dir, p.SFiles, p.CFiles),
			PkgPath:    pkgpath,
			Imports:    importMap(p.Imports),
		})
		if cfg.Tests {
			testID := fmt.Sprintf("%s [%s.test]", id, id)
			if len(p.TestGoFiles) > 0 || len(p.XTestGoFiles) > 0 {
				if isRoot {
					roots = append(roots, testID)
				}
				result = append(result, &rawPackage{
					ID:         testID,
					Name:       p.Name,
					GoFiles:    absJoin(p.Dir, p.GoFiles, p.TestGoFiles, p.CgoFiles),
					OtherFiles: absJoin(p.Dir, p.SFiles, p.CFiles),
					PkgPath:    pkgpath,
					Imports:    importMap(append(p.Imports, p.TestImports...)),
				})
			}
			if len(p.XTestGoFiles) > 0 {
				xtestID := fmt.Sprintf("%s_test [%s.test]", id, id)
				if isRoot {
					roots = append(roots, xtestID)
				}
				result = append(result, &rawPackage{
					ID:      xtestID,
					Name:    p.Name + "_test",
					GoFiles: absJoin(p.Dir, p.XTestGoFiles),
					PkgPath: pkgpath,
					Imports: importMap(append(p.XTestImports, testID)),
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

	buf, err := golist(ctx, cfg, golistargs_fallback(cfg, deps))
	if err != nil {
		return nil, nil, err
	}

	// Decode the JSON and convert it to rawPackage form.
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
func getDeps(ctx context.Context, cfg *rawConfig, words ...string) (originalSet map[string]*jsonPackage, deps []string, err error) {
	buf, err := golist(ctx, cfg, golistargs_fallback(cfg, words))
	if err != nil {
		return nil, nil, err
	}

	depsSet := make(map[string]bool)
	originalSet = make(map[string]*jsonPackage)

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

func golistargs_fallback(cfg *rawConfig, words []string) []string {
	fullargs := []string{"list", "-e", "-json"}
	fullargs = append(fullargs, cfg.Flags...)
	fullargs = append(fullargs, "--")
	fullargs = append(fullargs, words...)
	return fullargs
}
