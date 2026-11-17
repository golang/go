// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/imports"
	"cmd/go/internal/logos/capabilities"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
)

func checkLogosLoaded(ctx context.Context, ld *modload.Loader, all []*Package) {
	checkLogosModules(ctx, ld, all)
	checkLogosCapabilities(ld, all)
	noteLogosBuildModules(all)
	if cfg.Logos {
		// These checks run after load propagated its errors. PackageList is
		// in postorder, so each importer sees its dependencies' final state.
		// Keep each diagnostic on the package that owns it.
		for _, p := range all {
			for _, dep := range p.Internal.Imports {
				if dep.Incomplete {
					p.Incomplete = true
					break
				}
			}
		}
	}
}

func noteLogosBuildModules(all []*Package) {
	if !cfg.Logos {
		return
	}
	for _, p := range all {
		if p.Module != nil {
			capabilities.NoteBuildModule(p.Module.Path)
		}
	}
}

func checkLogosCapabilities(ld *modload.Loader, all []*Package) {
	if !cfg.Logos {
		return
	}
	cf, err := logosConfig(ld)
	sev := capabilities.Allow
	var allowlist []string
	if err == nil && cf != nil {
		sev = capabilities.Severity(cf.Capabilities.TransitiveUnsafe)
		allowlist = cf.Capabilities.TransitiveUnsafeAllowlist
	}
	if sev == capabilities.Allow {
		return
	}
	nodes := make([]capabilities.Node, 0, len(all))
	for _, p := range all {
		if p.Internal.TestmainGo != nil {
			continue
		}
		nodes = append(nodes, capNode(p))
	}
	violations := capabilities.Check(sev, allowlist, nodes)
	if len(violations) == 0 {
		return
	}
	byPath := make(map[string]*Package, len(all))
	for _, p := range all {
		if _, ok := byPath[p.ImportPath]; !ok {
			byPath[p.ImportPath] = p
		}
	}
	jsonOut := settings.Current().JSON
	if sev == capabilities.Warn {
		for _, v := range violations {
			if jsonOut {
				fmt.Fprintln(os.Stderr, capabilities.FormatJSON(v, "warn", capSourceFile(byPath[firstChain(v)])))
			} else {
				fmt.Fprintln(os.Stderr, capabilities.FormatWarning(v))
			}
		}
		return
	}
	msgsByCarrier := make(map[*Package][]string)
	for _, v := range violations {
		if jsonOut {
			fmt.Fprintln(os.Stderr, capabilities.FormatJSON(v, "deny", capSourceFile(byPath[firstChain(v)])))
		}
		// Blame the first chain package that does not already carry a
		// diagnostic, not the first package alphabetically; a package
		// with an existing error keeps it.
		var carrier *Package
		for _, path := range v.Chain {
			cand := byPath[path]
			if cand == nil {
				continue
			}
			if carrier == nil {
				carrier = cand
			}
			if cand.Error == nil {
				carrier = cand
				break
			}
		}
		if carrier == nil {
			carrier = firstMainPackage(all)
		}
		if carrier != nil && carrier.Error != nil {
			// A package that already failed load keeps its diagnostic;
			// report this violation inline so it is not hidden.
			if !jsonOut {
				fmt.Fprintln(os.Stderr, capabilities.FormatViolation(v))
			}
			continue
		}
		if carrier != nil {
			msgsByCarrier[carrier] = append(msgsByCarrier[carrier], capabilities.FormatViolation(v))
		}
	}
	for carrier, msgs := range msgsByCarrier {
		text := strings.Join(msgs, "\n")
		if jsonOut {
			// JSON consumers get the record lines above; the package
			// error stays one line so the chain/help block never
			// leaks into the JSON stream.
			if i := strings.IndexByte(text, '\n'); i >= 0 {
				text = text[:i]
			}
		}
		// A command-line package has an empty ImportStack, so
		// PackageError would drop the path prefix; put it in the
		// message instead.
		carrier.Error = &PackageError{Err: fmt.Errorf("%s: %s", carrier.ImportPath, text)}
		carrier.Incomplete = true
	}
}

// firstChain names the main-module package heading a violation's
// import chain, or "" for an empty chain.
func firstChain(v capabilities.Violation) string {
	if len(v.Chain) == 0 {
		return ""
	}
	return v.Chain[0]
}

// capSourceFile resolves p to a real source path for the Sec 14
// diagnostics schema: the first source file in Dir, or Dir itself when
// the package has none.
func capSourceFile(p *Package) string {
	if p == nil {
		return ""
	}
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		if len(files) > 0 {
			return filepath.Join(p.Dir, files[0])
		}
	}
	return p.Dir
}

func capNode(p *Package) capabilities.Node {
	n := capabilities.Node{
		ImportPath: p.ImportPath,
		Imports:    p.Imports,
		Goroot:     p.Goroot || p.Standard,
		Main:       capIsMain(p),
	}
	if n.Goroot || n.Main {
		return n
	}
	n.Imports = slices.Concat(p.Imports, capExcludedImports(p))
	n.Asm = capHoldsAssembly(p)
	return n
}

func capExcludedImports(p *Package) []string {
	var out []string
	for _, name := range p.IgnoredGoFiles {
		if !capIsTestFile(name) {
			out = append(out, capFileImports(filepath.Join(p.Dir, name))...)
		}
	}
	return out
}

func capIsTestFile(name string) bool {
	return strings.HasSuffix(strings.TrimSuffix(name, filepath.Ext(name)), "_test")
}

func capFileImports(file string) []string {
	fsys.Trace("capRead", file)
	f, err := fsys.Open(file)
	if err != nil {
		return nil
	}
	defer f.Close()
	var quoted []string
	head, err := imports.ReadImports(f, false, &quoted)
	if err != nil || !imports.ShouldBuild(head, imports.AnyTags()) {
		return nil
	}
	var out []string
	for _, q := range quoted {
		if path, err := strconv.Unquote(q); err == nil {
			out = append(out, path)
		}
	}
	return out
}

func capHoldsAssembly(p *Package) bool {
	if len(p.SFiles) > 0 || len(p.SysoFiles) > 0 {
		return true
	}
	return slices.ContainsFunc(p.IgnoredOtherFiles, func(name string) bool {
		return capIsObject(name) || capIsAssembly(name) && capBuildsAnywhere(filepath.Join(p.Dir, name))
	})
}

func capIsObject(name string) bool {
	return filepath.Ext(name) == ".syso"
}

func capIsAssembly(name string) bool {
	return slices.Contains([]string{".s", ".S", ".sx"}, filepath.Ext(name))
}

func capBuildsAnywhere(file string) bool {
	f, err := fsys.Open(file)
	if err != nil {
		return true
	}
	defer f.Close()
	head, err := io.ReadAll(io.LimitReader(f, 1<<16))
	return err != nil || imports.ShouldBuild(head, imports.AnyTags())
}

func capIsMain(p *Package) bool {
	if p.ImportPath == "command-line-arguments" {
		return true
	}
	return p.Module != nil && p.Module.Main
}

func firstMainPackage(all []*Package) *Package {
	var mains []*Package
	for _, p := range all {
		if p.Internal.TestmainGo != nil || !capIsMain(p) || p.Error != nil {
			continue
		}
		mains = append(mains, p)
	}
	if len(mains) == 0 {
		return nil
	}
	slices.SortFunc(mains, func(a, b *Package) int { return strings.Compare(a.ImportPath, b.ImportPath) })
	return mains[0]
}
