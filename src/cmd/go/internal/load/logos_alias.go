// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"internal/logosfile"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
)

const logosAliasRule = "portability::alias_extension"

var logosAliasAllow = regexp.MustCompile(`^//logos:allow\(portability::alias_extension\)(?:\s.*)?$`)

func checkLogosAliases(ld *modload.Loader, p *Package) {
	if !cfg.Logos || p.Internal.TestmainGo != nil {
		return
	}
	var failures []error
	for _, name := range logosAliasFiles(p) {
		level := logosAliasLevel(ld, p)
		allow := logosAliasDirective(filepath.Join(p.Dir, name))
		failures = append(failures, reportLogosAlias(name, level, allow)...)
	}
	if len(failures) > 0 && p.Error == nil {
		p.Error = &PackageError{Err: errors.Join(failures...)}
		p.Incomplete = true
	}
}

func logosAliasFiles(p *Package) []string {
	mod := p.Module
	if mod == nil || !mod.Main {
		return nil
	}
	elem, _, _ := strings.Cut(mod.Path, "/")
	if !strings.Contains(elem, ".") {
		return nil
	}
	var names []string
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range files {
			if logosfile.IsAlias(name) {
				names = append(names, name)
			}
		}
	}
	return names
}

func logosAliasLevel(ld *modload.Loader, p *Package) string {
	return settings.ModuleAt(logosConfigRootForPackage(ld, p)).Severity("portability", "warn")
}

func logosAliasDirective(path string) token.Position {
	src, err := fsys.ReadFile(path)
	if err != nil {
		return token.Position{}
	}
	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, path, src, parser.PackageClauseOnly|parser.ParseComments)
	if f == nil || f.Doc == nil {
		return token.Position{}
	}
	for _, c := range f.Doc.List {
		if logosAliasAllow.MatchString(c.Text) {
			return fset.Position(c.Slash)
		}
	}
	return token.Position{}
}

func reportLogosAlias(name, level string, allow token.Position) []error {
	text := fmt.Sprintf("[%s] %s: loGo/s alias extension file; rename to .lgo before publishing", logosAliasRule, name)
	switch {
	case level == "allow":
		return nil
	case level == "forbid":
		return forbiddenLogosAlias(name, text, allow)
	case allow.IsValid():
		return nil
	case level == "deny":
		return []error{errors.New(text)}
	}
	fmt.Fprintf(os.Stderr, "go: warning: %s\n", text)
	return nil
}

func forbiddenLogosAlias(name, text string, allow token.Position) []error {
	errs := []error{errors.New(text)}
	if allow.IsValid() {
		errs = append(errs, fmt.Errorf("[correctness::allow_forbidden_rule] %s:%d:%d: //logos:allow(%s): rule %s is forbid and cannot be allowed", name, allow.Line, allow.Column, logosAliasRule, logosAliasRule))
	}
	return errs
}
