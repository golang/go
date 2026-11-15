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
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/str"
)

type lintActor struct{}

func (b *Builder) LintAction(p *load.Package) *Action {
	if p.UsesCgo() || p.UsesSwig() {
		return b.CompileAction(ModeBuild, ModeBuild, p)
	}
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

func LintsPackage(p *load.Package) bool {
	return p.Internal.LogosLint != nil && (len(p.GoFiles) > 0 || p.UsesCgo() || p.UsesSwig()) &&
		p.Internal.TestmainGo == nil
}

func logosCompileIsLint(p *load.Package) bool {
	return settings.Current().LintCmd && LintsPackage(p) && logosGenerates(p) &&
		(p.Internal.CmdlinePkg || p.Internal.CmdlineFiles)
}

func logosCompileLintOwned(a *Action) bool {
	if _, ok := a.Actor.(lintActor); ok {
		return false
	}
	p := a.Package
	return settings.Current().LintCmd && LintsPackage(p) && !logosGenerates(p) &&
		(p.Internal.CmdlinePkg || p.Internal.CmdlineFiles)
}

func logosGenerates(p *load.Package) bool {
	return p.UsesCgo() || p.UsesSwig()
}

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

func logosCompilePolicy(a *Action) []byte {
	p := a.Package
	if p == nil || p.Internal.LogosLint == nil {
		return nil
	}
	if _, ok := a.Actor.(lintActor); ok {
		return p.Internal.LogosLint
	}
	if p.IsLogos() && !settings.Current().LintCmd {
		return p.Internal.LogosLint
	}
	if settings.Current().LintCmd && LintsPackage(p) && (!logosGenerates(p) || logosCompileIsLint(p)) {
		return p.Internal.LogosLint
	}
	return nil
}

func logosLintFlags(sh *Shell, a *Action) ([]string, error) {
	var flags []string
	if settings.Current().JSON {
		flags = append(flags, "-logos.json")
	}
	if cfg.Logos {
		flags = append(flags, "-logos.allocfacts")
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
	} else if logosCompileIsLint(a.Package) {
		flags = append(flags, "-logos.lint")
	} else if logosCompileLintOwned(a) {
		flags = append(flags, "-logos.lintowned")
	}
	return flags, nil
}

func logosGoFactsFlags(p *load.Package) []string {
	if p == nil {
		return nil
	}
	if cfg.Logos && !p.IsLogos() && !logosStandardProvenance(p) {
		return []string{"-logos.gofacts"}
	}
	return nil
}

func logosLintActionID(h io.Writer, a *Action) {
	if policy := logosCompilePolicy(a); policy != nil {
		fmt.Fprintf(h, "logos lint %x\n", sha256.Sum256(policy))
		if logosCompileIsLint(a.Package) {
			fmt.Fprintf(h, "logos lint compile\n")
		}
		if logosCompileLintOwned(a) {
			fmt.Fprintf(h, "logos lint owned\n")
		}
	}
	if settings.Current().JSON {
		fmt.Fprintf(h, "logos json\n")
	}
	if cfg.Logos {
		fmt.Fprintf(h, "logos allocfacts\n")
	}
	if len(logosGoFactsFlags(a.Package)) > 0 {
		fmt.Fprintf(h, "logos gofacts\n")
	}
}

func logosJSONLines(text string) bool {
	if !settings.Current().JSON || strings.TrimSpace(text) == "" {
		return false
	}
	for line := range strings.Lines(text) {
		if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "{") {
			return false
		}
	}
	return true
}
