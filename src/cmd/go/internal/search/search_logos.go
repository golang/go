// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package search

import (
	"go/build"
	"go/parser"
	"go/token"
	"internal/logosfile"
	"path/filepath"
	"strconv"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
)

// logosKeepCgoOnly reports whether a wildcard match must keep a directory
// whose ImportDir failed with NoGoError. In logos mode a package whose
// every admitted source file needs cgo holds those files in
// IgnoredGoFiles and fails with NoGoError, so the upstream "the package
// does not actually exist" skip would drop a cgo-only loGo/s package
// before loading runs and the cgo policy check could reject it with
// [restriction::cgo_usage]. Keeping the directory resolves its package
// path, and loading then produces the denial exactly as an explicit
// package path does.
//
// A file qualifies only when cgo is the sole reason it sits in
// IgnoredGoFiles. go/build's Context.MatchFile never applies the cgo
// rule itself (it checks the extension, the name-based OS/arch tags, and
// the //go:build constraints; the cgo classification lives in
// Context.ImportDir's file loop, fed by the parsed imports), so the
// decision needs both halves. MatchFile against a copy of the build
// context with CgoEnabled set rules out files excluded by a build tag or
// a name suffix, which cgo cannot override; the imports check rules out
// a //go:build cgo file that carries no "C" import, which cgo being off
// excludes without the package being a cgo package. A file rejected by
// either half keeps the upstream silent skip. Both halves judge an
// overlaid replacement by its effective content: MatchFile reads it
// through the build context's file hooks, and the imports probe reads
// through the same overlay-aware file system. Only loGo/s sources
// qualify: a plain go context never admits .lgo files into
// IgnoredGoFiles, so upstream behavior is unchanged.
func logosKeepCgoOnly(p *build.Package) bool {
	if p == nil || cfg.BuildContext.CgoEnabled {
		return false
	}
	cgoOn := cfg.BuildContext
	cgoOn.CgoEnabled = true
	for _, name := range p.IgnoredGoFiles {
		if !logosfile.Is(name) {
			continue
		}
		matched, err := cgoOn.MatchFile(p.Dir, name)
		if err != nil || !matched {
			continue
		}
		if logosFileImportsC(p.Dir, name) {
			return true
		}
	}
	return false
}

// logosFileImportsC reports whether the named Go-family source file in dir
// imports "C". It reads through cmd/go/internal/fsys, the overlay-aware
// file system MatchFile and loading read, so an overlaid replacement is
// judged by its effective content rather than the file on disk. A read or
// parse failure reads as no import, matching how loading treats
// unreadable files.
func logosFileImportsC(dir, name string) bool {
	path := filepath.Join(dir, name)
	src, err := fsys.ReadFile(path)
	if err != nil {
		return false
	}
	parsed, err := parser.ParseFile(token.NewFileSet(), path, src, parser.ImportsOnly)
	if err != nil || parsed == nil {
		return false
	}
	// ImportDir never admits a `package documentation` file, so cgo cannot be
	// what hides it.
	if parsed.Name.Name == "documentation" {
		return false
	}
	for _, imp := range parsed.Imports {
		if path, err := strconv.Unquote(imp.Path.Value); err == nil && path == "C" {
			return true
		}
	}
	return false
}
