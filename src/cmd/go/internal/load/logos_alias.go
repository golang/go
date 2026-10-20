// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modload"
)

// logosAliasRule is the rule ID of an alias-extension file.
const logosAliasRule = "portability::alias_extension"

// logosAliasAllow matches the //logos:allow directive that names the rule,
// optionally followed by a space and free text, as the compiler lint does.
var logosAliasAllow = regexp.MustCompile(`^//logos:allow\(portability::alias_extension\)(?:\s.*)?$`)

// logosIsCanonicalLgo reports whether name is a canonical .lgo file, after
// stripping trailing variation selectors the same way the shared filename
// predicate does.
func logosIsCanonicalLgo(name string) bool {
	return strings.HasSuffix(strings.TrimRight(name, "\uFE0E\uFE0F"), ".lgo")
}

// checkLogosAliases applies portability::alias_extension to one package
// after its imports have resolved (SPEC Sec 1.2, M9). The severity is the
// portability group of the logos.toml of the main module: warn prints
// "go: warning:" and the build goes on, deny and forbid fail the package
// with a package error, and allow is silent. A //logos:allow directive
// directly above the package clause of the file silences it under warn and
// deny; forbid ignores the directive and reports it as
// correctness::allow_forbidden_rule. An error already on the package keeps
// priority. The generated test main is exempt.
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

// logosAliasFiles returns the accepted alias-extension source files of p,
// the test lists included. Canonical .lgo files never count. Only packages
// in a main module whose path starts with a dotted element count: those are
// the modules that could be published, where the module zip rules reject
// alias extensions.
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
			if cfg.IsLogosFile(name) && !logosIsCanonicalLgo(name) {
				names = append(names, name)
			}
		}
	}
	return names
}

// logosAliasLevel returns the severity of the portability group for the
// main module of p: its logos.toml key, else the group default, warn. A
// module whose logos.toml does not parse takes the default, because
// checkLogosConfig reports the parse error.
func logosAliasLevel(ld *modload.Loader, p *Package) string {
	cf, err := logosConfigAt(logosConfigRootForPackage(ld, p))
	if err != nil || cf == nil {
		return "warn"
	}
	if level, ok := cf.Lints.Groups["portability"]; ok {
		return level
	}
	return "warn"
}

// logosAliasDirective returns the position of the //logos:allow directive
// that names the rule in the comment block directly above the package
// clause of the file at path, or an invalid position when there is none. A
// blank line detaches the block, as it does for a statement in the compiler.
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

// reportLogosAlias resolves one alias-extension file at level, given the
// position of its directive, which is invalid when the file has none. It
// prints a warning and returns the errors the file adds to its package.
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

// forbiddenLogosAlias returns the errors of an alias-extension file under
// forbid: the rule itself, and the directive when the file has one.
func forbiddenLogosAlias(name, text string, allow token.Position) []error {
	errs := []error{errors.New(text)}
	if allow.IsValid() {
		errs = append(errs, fmt.Errorf("[correctness::allow_forbidden_rule] %s:%d:%d: //logos:allow(%s): rule %s is forbid and cannot be allowed", name, allow.Line, allow.Column, logosAliasRule, logosAliasRule))
	}
	return errs
}
