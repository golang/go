// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
)

const (
	wasmOutputSuffix = ".wasm"
	wasmLoaderSuffix = ".loader.mjs"
)

// wantsWasmLoader reports the build-wide part of the loader decision (SPEC
// Sec 12.2, R-U1): the --wasm-loader flag is on, the build command is
// build, the target is GOARCH=wasm, and p is a main package. It reads the
// build configuration, never the module kind, so a Go main package and a
// loGo/s main package behave alike.
func wantsWasmLoader(p *Package) bool {
	return settings.Current().WasmLoader &&
		cfg.CmdName == "build" &&
		cfg.BuildContext.GOARCH == "wasm" &&
		p.Name == "main"
}

// isWasmFile reports whether out names a .wasm file.
func isWasmFile(out string) bool {
	return strings.HasSuffix(out, wasmOutputSuffix)
}

// EmitsWasmLoader reports at load time whether building p writes the host
// loader beside its output: the build-wide conditions hold and -o names a
// .wasm file. The load of p adds the implicit logos/rt import when it
// holds, so the exported helpers the loader calls are linked in. The
// default output name of a wasm main package carries no .wasm suffix, so
// a build without -o writes no loader.
func EmitsWasmLoader(p *Package) bool {
	return wantsWasmLoader(p) && isWasmFile(cfg.BuildO)
}

// WasmLoaderFor returns the loader file that goes beside the built output
// of p, and reports whether the build writes one: the build-wide
// conditions hold and the target of p is a .wasm file. The loader name is
// the target with its .wasm suffix replaced by .loader.mjs.
func WasmLoaderFor(p *Package) (string, bool) {
	if !wantsWasmLoader(p) || !isWasmFile(p.Target) {
		return "", false
	}
	return strings.TrimSuffix(p.Target, wasmOutputSuffix) + wasmLoaderSuffix, true
}

var loaderHelpers = []string{
	"_logos_scratch_alloc",
	"_logos_scratch_reset",
	"_logos_errmsg_ptr",
	"_logos_errmsg_len",
}

func pruneRedundantLogosRT(p *Package) {
	found := graphLoaderHelperSet(p)
	if len(found) == 0 {
		return
	}
	if len(found) == len(loaderHelpers) {
		p.Internal.Imports = dropLogosRTImport(p.Internal.Imports)
		p.Internal.CompiledImports = dropLogosRTCompiledImport(p.Internal.CompiledImports)
		return
	}
	var missing []string
	for _, name := range loaderHelpers {
		if !found[name] {
			missing = append(missing, name)
		}
	}
	p.Error = &PackageError{Err: fmt.Errorf("module graph provides some but not all wasm loader helpers: missing %s", strings.Join(missing, ", "))}
	p.Incomplete = true
}

func graphLoaderHelperSet(p *Package) map[string]bool {
	found := make(map[string]bool)
	seen := make(map[*Package]bool)
	stack := append([]*Package{p}, p.Internal.Imports...)
	for len(stack) > 0 {
		q := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if q == nil || seen[q] {
			continue
		}
		seen[q] = true
		if q.ImportPath == LogosRTPkg || q.Standard {
			continue
		}
		collectPackageWasmLoaderHelpers(q, found)
		if len(found) == len(loaderHelpers) {
			break
		}
		stack = append(stack, q.Internal.Imports...)
	}
	return found
}

func collectPackageWasmLoaderHelpers(q *Package, found map[string]bool) {
	for _, f := range q.GoFiles {
		data, err := os.ReadFile(filepath.Join(q.Dir, f))
		if err != nil {
			continue
		}
		for _, name := range fileWasmExportNames(data) {
			if slices.Contains(loaderHelpers, name) {
				found[name] = true
			}
		}
	}
}

func fileWasmExportNames(data []byte) []string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", data, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil
	}
	var names []string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || fn.Doc == nil {
			continue
		}
		for _, comment := range fn.Doc.List {
			if !strings.HasPrefix(comment.Text, "//go:wasmexport ") {
				continue
			}
			pos := fset.PositionFor(comment.Pos(), false)
			if len(bytes.TrimSpace(data[pos.Offset-pos.Column+1:pos.Offset])) != 0 {
				continue
			}
			fields := strings.Fields(comment.Text)
			if len(fields) == 2 {
				names = append(names, fields[1])
			}
		}
	}
	return names
}

func dropLogosRTImport(pkgs []*Package) []*Package {
	kept := make([]*Package, 0, len(pkgs))
	for _, q := range pkgs {
		if q == nil || q.ImportPath == LogosRTPkg {
			continue
		}
		kept = append(kept, q)
	}
	return kept
}

func dropLogosRTCompiledImport(paths []string) []string {
	kept := make([]string, 0, len(paths))
	for _, s := range paths {
		if s == LogosRTPkg {
			continue
		}
		kept = append(kept, s)
	}
	return kept
}
