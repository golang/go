// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

// This file holds the lint half of the build (M9): the lint action that
// logos lint runs for each main-module package, the -logos.cfg and -logos.json
// compiler flags, and the cache key of the lint policy. Plain go never
// reaches any of it: no package carries a policy and cfg.LogosJSON stays
// false.

// lintActor compiles one package with -logos.lintonly: the compiler
// type-checks the sources against the export data of the imports, lints
// them through escape analysis, and stops before SSA, so no object is
// written. Go and loGo/s packages take the same path.
type lintActor struct{}

// LintAction returns the action that lints package p. It depends on the
// ordinary compile of each import, for export data, and never on a compile
// of p itself.
func (b *Builder) LintAction(p *load.Package) *Action {
	return b.cacheAction("logos lint", p, func() *Action {
		a := &Action{
			Mode:    "lint",
			Package: p,
			Actor:   lintActor{},
			Objdir:  b.NewObjdir(),
		}
		for _, p1 := range p.Internal.Imports {
			a.Deps = append(a.Deps, b.CompileAction(ModeBuild, ModeBuild, p1))
		}
		return a
	})
}

// LintsPackage reports whether logos lint covers p: a package of a main
// module with its Go or loGo/s sources and no cgo, which the lint compile
// cannot reproduce without the generated files.
func LintsPackage(p *load.Package) bool {
	return p.Internal.LogosLint != nil && len(p.GoFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() &&
		p.Internal.TestmainGo == nil
}

// Act implements Actor.
func (lintActor) Act(b *Builder, ctx context.Context, a *Action) error {
	p := a.Package
	sh := b.Shell(a)
	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}
	embed, err := lintEmbedConfig(p)
	if err != nil {
		return err
	}
	icfg := lintImportConfig(a)
	_, out, err := BuildToolchain.gc(b, a, "", icfg, embed, "", false, "", str.StringList(p.GoFiles))
	return sh.reportCmd("", "", out, err)
}

// lintImportConfig returns the import config of the lint compile: the
// import maps of p and the export data of every dependency, as build does,
// plus the transitive packages a loGo/s package may synthesize imports of.
func lintImportConfig(a *Action) []byte {
	p := a.Package
	var icfg bytes.Buffer
	fmt.Fprintf(&icfg, "# import config\n")
	for i, raw := range p.Internal.RawImports {
		if final := p.Imports[i]; final != raw {
			fmt.Fprintf(&icfg, "importmap %s=%s\n", raw, final)
		}
	}
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil || p1.ImportPath == "" || a1.built == "" {
			continue
		}
		fmt.Fprintf(&icfg, "packagefile %s=%s\n", p1.ImportPath, a1.built)
	}
	logosSynthImportCfg(&icfg, a)
	return icfg.Bytes()
}

// lintEmbedConfig returns the embed config of p, or nil when p embeds
// nothing.
func lintEmbedConfig(p *load.Package) ([]byte, error) {
	if len(p.Internal.Embed) == 0 {
		return nil, nil
	}
	var embed struct {
		Patterns map[string][]string
		Files    map[string]string
	}
	embed.Patterns = p.Internal.Embed
	embed.Files = make(map[string]string)
	for _, file := range p.EmbedFiles {
		embed.Files[file] = fsys.Actual(filepath.Join(p.Dir, file))
	}
	js, err := json.Marshal(&embed)
	if err != nil {
		return nil, fmt.Errorf("marshal embedcfg: %v", err)
	}
	return js, nil
}

// logosCompilePolicy returns the lint policy action a passes to the compiler,
// or nil. The lint action always passes it. An ordinary compile passes it
// for a loGo/s package, whose lints run on every compile, except during
// logos lint, where the lint action covers the package.
func logosCompilePolicy(a *Action) []byte {
	p := a.Package
	if p == nil || p.Internal.LogosLint == nil {
		return nil
	}
	if _, ok := a.Actor.(lintActor); ok {
		return p.Internal.LogosLint
	}
	if p.IsLogos() && !cfg.LogosLintCmd {
		return p.Internal.LogosLint
	}
	return nil
}

// logosLintFlags returns the loGo/s lint flags of the compile of action a:
// -logos.json under --json-diagnostics, and for a linted package the
// policy file, written into the action work directory, and -logos.lintonly
// for the lint action. It is the hook line in (gcToolchain).gc.
func logosLintFlags(sh *Shell, a *Action) ([]string, error) {
	var flags []string
	if cfg.LogosJSON {
		flags = append(flags, "-logos.json")
	}
	policy := logosCompilePolicy(a)
	if policy == nil {
		return flags, nil
	}
	file := a.Objdir + "logos.cfg.json"
	if err := sh.writeFile(file, policy); err != nil {
		return nil, err
	}
	flags = append(flags, "-logos.cfg="+file)
	if _, ok := a.Actor.(lintActor); ok {
		flags = append(flags, "-logos.lintonly")
	}
	return flags, nil
}

// logosLintActionID adds the lint policy and the diagnostic format of a
// build action to its action ID. A change of logos.toml or of --pedantic
// recompiles a loGo/s package, and so does a switch to or from
// --json-diagnostics for a package of either kind, because -logos.json
// reaches every compile: the go command caches the compile output of a
// package and replays it on a cache hit, so text cached by one format must
// not answer a build in the other. It is the hook line in buildActionID.
func logosLintActionID(h io.Writer, a *Action) {
	if policy := logosCompilePolicy(a); policy != nil {
		fmt.Fprintf(h, "logos lint %x\n", sha256.Sum256(policy))
	}
	if cfg.LogosJSON {
		fmt.Fprintf(h, "logos json\n")
	}
}

// logosJSONLines reports whether text is nothing but JSON diagnostic lines
// under --json-diagnostics. Such output prints without the "# package"
// header line, so every line of stderr is one JSON object.
func logosJSONLines(text string) bool {
	if !cfg.LogosJSON || strings.TrimSpace(text) == "" {
		return false
	}
	for line := range strings.Lines(text) {
		if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "{") {
			return false
		}
	}
	return true
}
