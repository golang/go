// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"fmt"
	"os"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/capabilities"
	"cmd/go/internal/modload"
)

// checkLogosLoaded runs the whole load-census hook for logos mode: module
// purity (checkLogosModules) and the capabilities audit below. Both exits
// of PackagesAndErrors call it, so file-argument and pattern builds are
// policed identically.
func checkLogosLoaded(ld *modload.Loader, all []*Package) {
	checkLogosModules(all)
	checkLogosCapabilities(ld, all)
}

// checkLogosCapabilities enforces the logos.toml [capabilities]
// transitive_unsafe policy over the loaded package graph (M12-S1). It runs
// after imports resolve, beside checkLogosModules, so the check sees every
// module in the build. Plain go never reaches past the logos guard.
//
// A non-standard, non-main-module package that imports "unsafe" and matches
// no transitive_unsafe_allowlist glob is a violation with the shortest
// import chain from a main-module package. Severity deny attaches one error
// carrying every violation to the first error-free main-module package;
// warn prints each violation and lets the build continue; allow skips the
// walk. Config parse failures are left to checkLogosConfig, which already
// reported them; an unreadable policy fails closed with deny and no
// allowlists.
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
		nodes = append(nodes, capabilities.Node{
			ImportPath: p.ImportPath,
			Imports:    p.Imports,
			Goroot:     p.Goroot || p.Standard,
			Main:       capIsMain(p),
		})
	}
	violations := capabilities.CheckUnsafe(sev, allowlist, nodes)
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

// capIsMain reports whether p is first-party for the audit: a main-module
// package or the module-less command-line-arguments file list. Workspace
// members count as main modules, so a workspace member importing "unsafe"
// answers to its own module instead; use a replace-based dependency to
// police a transitive import.
func capIsMain(p *Package) bool {
	if p.ImportPath == "command-line-arguments" {
		return true
	}
	return p.Module != nil && p.Module.Main
}

// firstMainPackage returns the lexicographically first error-free
// main-module package, or nil when every such package already has an
// error. The capabilities diagnostic lands there so it never overwrites a
// more specific error.
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
