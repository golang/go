// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package list

import (
	"encoding/json"
	"fmt"
	"internal/logosfile"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/load"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/logos/settings"
	"cmd/internal/logos/editmap"
)

var logosStage string

type logosLoweredFile struct {
	File         string          `json:",omitempty"`
	LogosLowered string          `json:",omitempty"`
	LogosEditMap json.RawMessage `json:",omitempty"`
}

func init() {
	if cfg.Logos {
		CmdList.Flag.BoolVar(&settings.CommandLine.JSON, "json-diagnostics", false, "")
	}
}

func logosArmCompiled() {
	if !cfg.Logos || !*listCompiled {
		return
	}
	dir, err := os.MkdirTemp("", "logoslist")
	if err != nil {
		return
	}
	logosStage = dir
	base.AtExit(func() {
		load.DisarmLogosEject()
		os.RemoveAll(dir)
	})
	load.ArmLogosEject(dir)
}

func logosListPackage(p *load.Package) any {
	if logosStage == "" || p.Standard {
		return &p.PackagePublic
	}
	out := &logosPackage{PackagePublic: &p.PackagePublic}
	out.LogosFiles = logosFiles(p, logosStage)
	if !p.DepOnly {
		out.LogosHints = logosHints(p, logosStage)
	}
	if p.Error != nil {
		for line := range strings.SplitSeq(p.Error.Err.Error(), "\n") {
			if strings.HasPrefix(line, "{") {
				out.LogosDiags = append(out.LogosDiags, json.RawMessage(line))
			}
		}
	}
	return out
}

type logosPackage struct {
	*load.PackagePublic

	LogosFiles []logosLoweredFile `json:",omitempty"`
	LogosHints json.RawMessage    `json:",omitempty"`
	LogosDiags []json.RawMessage  `json:",omitempty"`
}

func logosFillCompiled(pkgs []*load.Package) {
	if logosStage == "" || !cfg.Logos || !*listCompiled {
		return
	}
	exportsOf = make(map[string]string, len(pkgs))
	for _, q := range pkgs {
		exportsOf[q.ImportPath] = q.Export
	}
	overlayReplaces = nil
	if data, err := os.ReadFile(fsys.OverlayFile); err == nil {
		var ov struct {
			Replace map[string]string `json:"Replace"`
		}
		if json.Unmarshal(data, &ov) == nil {
			overlayReplaces = ov.Replace
		}
	}
}

var exportsOf map[string]string

var overlayReplaces map[string]string

func logosFiles(p *load.Package, stage string) []logosLoweredFile {
	dir := filepath.Join(stage, filepath.FromSlash(p.ImportPath))
	var out []logosLoweredFile
	for _, name := range p.GoFiles {
		if !logosfile.Is(name) {
			continue
		}
		lowered := filepath.Join(dir, logosEjectedName(name))
		text, err := os.ReadFile(lowered)
		if err != nil {
			continue
		}
		raw, err := os.ReadFile(lowered + editmap.Suffix)
		if err != nil {
			continue
		}
		out = append(out, logosLoweredFile{File: name, LogosLowered: string(text), LogosEditMap: json.RawMessage(raw)})
	}
	return out
}

func logosEjectedName(name string) string {
	return strings.TrimSuffix(name, filepath.Ext(name)) + ".go"
}

func logosHints(p *load.Package, stage string) json.RawMessage {
	exe, err := os.Executable()
	if err != nil {
		return nil
	}
	work, err := os.MkdirTemp(stage, "hints")
	if err != nil {
		return nil
	}
	importcfg := filepath.Join(work, "importcfg")
	if err := os.WriteFile(importcfg, logosImportCfg(p), 0o666); err != nil {
		return nil
	}
	sources, ok := logosHintSources(p)
	if !ok || len(sources) == 0 {
		return nil
	}
	out := filepath.Join(work, "hints.json")
	args := []string{"tool", "compile", "-p", p.ImportPath, "-importcfg", importcfg,
		"-logos.mode=hints", "-logos.hintsout=" + out, "-o", filepath.Join(work, "x.a")}
	cmd := exec.Command(exe, append(args, sources...)...)
	cmd.Args[0] = "logos"
	cmd.Dir = p.Dir
	cmd.Env = logosHintsEnv(p)
	if err := cmd.Run(); err != nil {
		return nil
	}
	data, err := os.ReadFile(out)
	if err != nil {
		return nil
	}
	return json.RawMessage(data)
}

func logosImportCfg(p *load.Package) []byte {
	var buf []byte
	for _, imp := range p.Imports {
		if e := exportsOf[imp]; e != "" {
			buf = append(buf, fmt.Sprintf("packagefile %s=%s\n", imp, e)...)
		}
	}
	return buf
}

func logosHintSources(p *load.Package) ([]string, bool) {
	files := make([]string, 0, len(p.GoFiles))
	for _, name := range p.GoFiles {
		path := filepath.Join(p.Dir, name)
		if back, ok := overlayReplaces[path]; ok {
			path = back
		}
		if _, err := os.Stat(path); err != nil {
			return nil, false
		}
		files = append(files, path)
	}
	return files, true
}

func logosHintsEnv(p *load.Package) []string {
	overrides := map[string]string{
		"GOEXPERIMENT": cfg.Getenv("GOEXPERIMENT"),
		"CGO_ENABLED":  cfg.Getenv("CGO_ENABLED"),
	}
	amd64, arm64 := config.ArchLevels(logosTargetCPU(p))
	if v := cfg.Getenv("GOAMD64"); v != "" {
		amd64 = v
	}
	if v := cfg.Getenv("GOARM64"); v != "" {
		arm64 = v
	}
	if os.Getenv("GOAMD64") == "" {
		overrides["GOAMD64"] = amd64
	}
	if os.Getenv("GOARM64") == "" {
		overrides["GOARM64"] = arm64
	}
	env := make([]string, 0, len(os.Environ())+len(overrides))
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if _, drop := overrides[k]; drop {
			continue
		}
		env = append(env, kv)
	}
	for _, k := range slices.Sorted(maps.Keys(overrides)) {
		env = append(env, k+"="+overrides[k])
	}
	return env
}

func logosTargetCPU(p *load.Package) string {
	if p.Module == nil || p.Module.Dir == "" {
		return ""
	}
	m := settings.ModuleAt(p.Module.Dir)
	if m.Err != nil || m.File == nil {
		return ""
	}
	return m.File.Build.TargetCPU
}
