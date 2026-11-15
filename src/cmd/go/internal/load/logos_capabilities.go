// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
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
	"cmd/go/internal/modload"
)

func checkLogosLoaded(ld *modload.Loader, all []*Package) {
	checkLogosModules(all)
	checkLogosCapabilities(ld, all)
	noteLogosBuildModules(all)
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
	sev := capabilities.Deny
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
	if sev == capabilities.Warn {
		for _, v := range violations {
			fmt.Fprintln(os.Stderr, capabilities.FormatWarning(v))
		}
		return
	}
	var msgs []string
	for _, v := range violations {
		msgs = append(msgs, capabilities.FormatViolation(v))
	}
	if carrier := firstMainPackage(all); carrier != nil {
		carrier.Error = &PackageError{Err: fmt.Errorf("%s", strings.Join(msgs, "\n"))}
		carrier.Incomplete = true
	}
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
