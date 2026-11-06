// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

// Analysis in go test, which replaces go test's use of cmd/vet when
// enabled by GOEXPERIMENT=vetfiles (see cfg.Experiment.VetFiles).

import (
	"context"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// positiveRefs holds the references that enable analyzers for some
// root package.
var positiveRefs = make(map[*load.AnalysisRef]bool)

// vetAnalysisConfig returns go test's analysis configuration for
// package p, from the -vet flag: the selected members of the vet
// suite, or all of them for -vet=all.
//
// As with cmd/vet, packages in GOROOT are by default checked with
// nearly all analyzers (see gorootAnalysisConfig).
func vetAnalysisConfig(p *load.Package) *load.AnalysisConfig {
	if p.Goroot && !testVet.explicit {
		return gorootAnalysisConfig()
	}
	return userAnalysisConfig()
}

var userAnalysisConfig = sync.OnceValue(func() *load.AnalysisConfig {
	if testVet.explicit && testVet.flags == nil {
		return load.VetAnalysisConfig(nil) // -vet=all
	}
	var names []string
	for _, flag := range testVet.flags {
		names = append(names, strings.TrimPrefix(flag, "-"))
	}
	return load.VetAnalysisConfig(names)
})

// gorootAnalysisConfig is the default configuration for packages in
// GOROOT: all vet analyzers except unsafeptr (there is too much
// unsafe.Pointer code that it does not like in low-level packages)
// and unreachable (during testing it is common to make changes, such
// as forced returns, that make code unreachable). See also
// work.vetFlags.
var gorootAnalysisConfig = sync.OnceValue(func() *load.AnalysisConfig {
	config := load.VetAnalysisConfig(nil)
	// TODO(adonovan): this list is temporary; it will soon move
	// into a go.vet file in GOROOT/src.
	for _, name := range []string{"unsafeptr", "unreachable"} {
		config.Negative = append(config.Negative, &load.AnalysisRef{Negative: true, Name: name})
	}
	return config
})

// addTestAnalysis adds an action to report the analyzers enabled for
// root package p by go test's analysis configuration.
//
// As with cmd/vet before it, the test's run action depends on
// the report, so the test does not run if the analysis fails. The
// install action, if any, also depends on it.
func addTestAnalysis(ld *modload.Loader, b *work.Builder, p *load.Package, runAction, installAction *work.Action) {
	if testVet.off || testC {
		// -vet=off, or go test -c, which is a build, not a test.
		// Building code must not execute it (go.dev/blog/supply-chain),
		// and the analyzers declared by configuration files, as
		// proposed in go.dev/issue/82464, may come from any module,
		// so -c runs no analysis: for consistency, not even go test's
		// vet checks. (This differs from cmd/vet, which -c ran.)
		return
	}
	config := vetAnalysisConfig(p)
	for _, a := range config.Positive {
		positiveRefs[a] = true
	}
	a := b.AnalysisVetAction(ld, p, config)
	// Install will clean the build directory; make sure analysis runs first.
	if installAction != nil {
		installAction.Deps = append(installAction.Deps, a)
	}
	runAction.Deps = append(runAction.Deps, a)
}

// buildAnalysisTool arranges to build the analysis tool containing
// the variables referenced by the configurations of the root
// packages, and to query its metadata.
//
// The tool is built in this process, in the tests' build
// configuration (whatever their GOOS, GOARCH, build tags, -race,
// -gcflags, and so on), sharing compiled packages with the tests, and
// it runs on the host, as do the tests. (When the tests run under an
// exec wrapper, go test uses cmd/vet instead; see analysisUsesCmdVet.)
func buildAnalysisTool(ld *modload.Loader, ctx context.Context, b *work.Builder, opts load.PackageOpts) {
	if !b.HasAnalysisActions() {
		return
	}
	refs := slices.SortedFunc(maps.Keys(positiveRefs), load.CompareAnalysisRefs)
	for _, err := range synthesizeAnalysisTool(ld, ctx, b, opts, refs) {
		base.Errorf("go: %v", err)
	}
}

// analysisUsesCmdVet reports whether go test uses cmd/vet in place of
// the analysis tool because the tests run under an exec wrapper:
// either the program named by the -exec flag, or the
// go_$GOOS_$GOARCH_exec program that go test finds in $PATH when the
// target platform is not the host's (see work.FindExecCmd).
//
// The analysis tool is built in the tests' configuration and runs on
// the host. A wrapper indicates that the tests do not run directly on
// the host but in an emulator, device, virtual machine, or other
// sandbox, which may be a security boundary that running the tool on
// the host would bypass; and the tool, built for the target, might
// not run on the host anyway. So go test does not run the tool, and
// instead uses cmd/vet, which is built for the host and runs only the
// analyzers of the standard library. The analysis tool is never
// cross-compiled for the host.
var analysisUsesCmdVet = sync.OnceValue(func() bool {
	return len(work.FindExecCmd()) > 0
})

// synthesizeAnalysisTool synthesizes the main package of the analysis
// tool, containing the variables referenced by refs, and adds the
// actions to build it and query its metadata to b. It returns errors
// loading the packages of the references.
func synthesizeAnalysisTool(ld *modload.Loader, ctx context.Context, b *work.Builder, opts load.PackageOpts, refs []*load.AnalysisRef) []error {
	pmain, src, errs := load.AnalysisToolPackage(ld, ctx, opts, refs)
	dir := b.CompileAction(work.ModeBuild, work.ModeBuild, pmain).Objdir
	if err := b.BackgroundShell().Mkdir(dir); err != nil {
		base.Fatal(err)
	}
	pmain.Dir = dir
	if !cfg.BuildN {
		if err := os.WriteFile(filepath.Join(dir, pmain.GoFiles[0]), src, 0666); err != nil {
			base.Fatal(err)
		}
	}
	link := b.LinkAction(ld, work.ModeBuild, work.ModeBuild, pmain)
	link.Target = dir + "analysistool" + cfg.ExeSuffix
	link.CacheExecutable = true
	b.SetAnalysisTool(link)
	return errs
}
