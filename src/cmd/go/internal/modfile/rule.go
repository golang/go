// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

type File struct {
	Module  *Module
	Require []*Require
	Exclude []*Exclude
	Replace []*Replace

	Syntax *FileSyntax
}

type Module struct {
	Mod   module.Version
	Major string
}

type Require struct {
	Mod    module.Version
	Syntax *Line
}

type Exclude struct {
	Mod    module.Version
	Syntax *Line
}

type Replace struct {
	Old module.Version
	New module.Version

	Syntax *Line
}

func (f *File) AddModuleStmt(path string) {
	f.Module = &Module{
		Mod: module.Version{Path: path},
	}
	if f.Syntax == nil {
		f.Syntax = new(FileSyntax)
	}
	f.Syntax.Stmt = append(f.Syntax.Stmt, &Line{
		Token: []string{"module", AutoQuote(path)},
	})
}

func (f *File) AddComment(text string) {
	if f.Syntax == nil {
		f.Syntax = new(FileSyntax)
	}
	f.Syntax.Stmt = append(f.Syntax.Stmt, &CommentBlock{
		Comments: Comments{
			Before: []Comment{
				{
					Token: text,
				},
			},
		},
	})
}

type VersionFixer func(path, version string) (string, error)

func Parse(file string, data []byte, fix VersionFixer) (*File, error) {
	fs, err := parse(file, data)
	if err != nil {
		return nil, err
	}
	f := &File{
		Syntax: fs,
	}

	var errs bytes.Buffer
	for _, x := range fs.Stmt {
		switch x := x.(type) {
		case *Line:
			f.add(&errs, x, x.Token[0], x.Token[1:], fix)

		case *LineBlock:
			if len(x.Token) > 1 {
				fmt.Fprintf(&errs, "%s:%d: unknown block type: %s\n", file, x.Start.Line, strings.Join(x.Token, " "))
				continue
			}
			switch x.Token[0] {
			default:
				fmt.Fprintf(&errs, "%s:%d: unknown block type: %s\n", file, x.Start.Line, strings.Join(x.Token, " "))
				continue
			case "module", "require", "exclude", "replace":
				for _, l := range x.Line {
					f.add(&errs, l, x.Token[0], l.Token, fix)
				}
			}
		}
	}

	if errs.Len() > 0 {
		return nil, errors.New(strings.TrimRight(errs.String(), "\n"))
	}
	return f, nil
}

func (f *File) add(errs *bytes.Buffer, line *Line, verb string, args []string, fix VersionFixer) {
	// TODO: We should pass in a flag saying whether this module is a dependency.
	// If so, we should ignore all unknown directives and not attempt to parse
	// replace and exclude either. They don't matter, and it will work better for
	// forward compatibility if we can depend on modules that have local changes.

	// TODO: For the target module (not dependencies), maybe we should
	// relax the semver requirement and rewrite the file with updated info
	// after resolving any versions. That would let people type commit hashes
	// or tags or branch names, and then vgo would fix them.

	switch verb {
	default:
		fmt.Fprintf(errs, "%s:%d: unknown directive: %s\n", f.Syntax.Name, line.Start.Line, verb)
	case "module":
		if f.Module != nil {
			fmt.Fprintf(errs, "%s:%d: repeated module statement\n", f.Syntax.Name, line.Start.Line)
			return
		}
		f.Module = new(Module)
		if len(args) != 1 {

			fmt.Fprintf(errs, "%s:%d: usage: module module/path [version]\n", f.Syntax.Name, line.Start.Line)
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		f.Module.Mod = module.Version{Path: s}
	case "require", "exclude":
		if len(args) != 2 {
			fmt.Fprintf(errs, "%s:%d: usage: %s module/path v1.2.3\n", f.Syntax.Name, line.Start.Line, verb)
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		old := args[1]
		v, err := parseVersion(s, &args[1], fix)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid module version %q: %v\n", f.Syntax.Name, line.Start.Line, old, err)
			return
		}
		v1, err := moduleMajorVersion(s)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		if v2 := semver.Major(v); v1 != v2 && (v1 != "v1" || v2 != "v0") {
			fmt.Fprintf(errs, "%s:%d: invalid module: %s should be %s, not %s (%s)\n", f.Syntax.Name, line.Start.Line, s, v1, v2, v)
			return
		}
		if verb == "require" {
			f.Require = append(f.Require, &Require{
				Mod:    module.Version{Path: s, Version: v},
				Syntax: line,
			})
		} else {
			f.Exclude = append(f.Exclude, &Exclude{
				Mod:    module.Version{Path: s, Version: v},
				Syntax: line,
			})
		}
	case "replace":
		if len(args) < 4 || len(args) > 5 || args[2] != "=>" {
			fmt.Fprintf(errs, "%s:%d: usage: %s module/path v1.2.3 => other/module v1.4\n\t or %s module/path v1.2.3 => ../local/directory", f.Syntax.Name, line.Start.Line, verb, verb)
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		old := args[1]
		v, err := parseVersion(s, &args[1], fix)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid module version %v: %v\n", f.Syntax.Name, line.Start.Line, old, err)
			return
		}
		v1, err := moduleMajorVersion(s)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		if v2 := semver.Major(v); v1 != v2 && (v1 != "v1" || v2 != "v0") {
			fmt.Fprintf(errs, "%s:%d: invalid module: %s should be %s, not %s (%s)\n", f.Syntax.Name, line.Start.Line, s, v1, v2, v)
			return
		}
		ns, err := parseString(&args[3])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		nv := ""
		if len(args) == 4 {
			if !isDirectoryPath(ns) {
				fmt.Fprintf(errs, "%s:%d: replacement module without version must be directory path (rooted or starting with ./ or ../)", f.Syntax.Name, line.Start.Line)
				return
			}
			if filepath.Separator == '/' && strings.Contains(ns, `\`) {
				fmt.Fprintf(errs, "%s:%d: replacement directory appears to be Windows path (on a non-windows system)", f.Syntax.Name, line.Start.Line)
				return
			}
		}
		if len(args) == 5 {
			old := args[4]
			nv, err = parseVersion(ns, &args[4], fix)
			if err != nil {
				fmt.Fprintf(errs, "%s:%d: invalid module version %v: %v\n", f.Syntax.Name, line.Start.Line, old, err)
				return
			}
			if isDirectoryPath(ns) {
				fmt.Fprintf(errs, "%s:%d: replacement module directory path %q cannot have version", f.Syntax.Name, line.Start.Line, ns)
				return
			}
		}
		// TODO: More sanity checks about directories vs module paths.
		f.Replace = append(f.Replace, &Replace{
			Old:    module.Version{Path: s, Version: v},
			New:    module.Version{Path: ns, Version: nv},
			Syntax: line,
		})
	}
}

func isDirectoryPath(ns string) bool {
	// Because go.mod files can move from one system to another,
	// we check all known path syntaxes, both Unix and Windows.
	return strings.HasPrefix(ns, "./") || strings.HasPrefix(ns, "../") || strings.HasPrefix(ns, "/") ||
		strings.HasPrefix(ns, `.\`) || strings.HasPrefix(ns, `..\`) || strings.HasPrefix(ns, `\`) ||
		len(ns) >= 2 && ('A' <= ns[0] && ns[0] <= 'Z' || 'a' <= ns[0] && ns[0] <= 'z') && ns[1] == ':'
}

func mustQuote(t string) bool {
	for _, r := range t {
		if !unicode.IsPrint(r) || r == ' ' || r == '"' || r == '\'' || r == '`' {
			return true
		}
	}
	return t == "" || strings.Contains(t, "//") || strings.Contains(t, "/*")
}

// AutoQuote returns s or, if quoting is required for s to appear in a go.mod,
// the quotation of s.
func AutoQuote(s string) string {
	if mustQuote(s) {
		return strconv.Quote(s)
	}
	return s
}

func parseString(s *string) (string, error) {
	t := *s
	if strings.HasPrefix(t, `"`) {
		var err error
		if t, err = strconv.Unquote(t); err != nil {
			return "", err
		}
	} else if strings.ContainsAny(t, "\"'`") {
		// Other quotes are reserved both for possible future expansion
		// and to avoid confusion. For example if someone types 'x'
		// we want that to be a syntax error and not a literal x in literal quotation marks.
		return "", fmt.Errorf("unquoted string cannot contain quote")
	}
	*s = AutoQuote(t)
	return t, nil
}

func parseVersion(path string, s *string, fix VersionFixer) (string, error) {
	t, err := parseString(s)
	if err != nil {
		return "", err
	}
	if fix != nil {
		var err error
		t, err = fix(path, t)
		if err != nil {
			return "", err
		}
	}
	if semver.IsValid(t) {
		*s = semver.Canonical(t)
		return *s, nil
	}
	return "", fmt.Errorf("version must be of the form v1.2.3")
}

func moduleMajorVersion(p string) (string, error) {
	if _, _, major, _, ok := ParseGopkgIn(p); ok {
		return major, nil
	}

	start := strings.LastIndex(p, "/") + 1
	v := p[start:]
	if !isMajorVersion(v) {
		return "v1", nil
	}
	if v[1] == '0' || v == "v1" {
		return "", fmt.Errorf("module path has invalid version number %s", v)
	}
	return v, nil
}

func isMajorVersion(v string) bool {
	if len(v) < 2 || v[0] != 'v' {
		return false
	}
	for i := 1; i < len(v); i++ {
		if v[i] < '0' || '9' < v[i] {
			return false
		}
	}
	return true
}

func (f *File) Format() ([]byte, error) {
	return Format(f.Syntax), nil
}

func (x *File) AddRequire(path, vers string) {
	var syntax *Line

	for i, stmt := range x.Syntax.Stmt {
		switch stmt := stmt.(type) {
		case *LineBlock:
			if len(stmt.Token) > 0 && stmt.Token[0] == "require" {
				syntax = &Line{Token: []string{AutoQuote(path), vers}}
				stmt.Line = append(stmt.Line, syntax)
				goto End
			}
		case *Line:
			if len(stmt.Token) > 0 && stmt.Token[0] == "require" {
				stmt.Token = stmt.Token[1:]
				syntax = &Line{Token: []string{AutoQuote(path), vers}}
				x.Syntax.Stmt[i] = &LineBlock{
					Comments: stmt.Comments,
					Token:    []string{"require"},
					Line: []*Line{
						stmt,
						syntax,
					},
				}
				goto End
			}
		}
	}

	syntax = &Line{Token: []string{"require", AutoQuote(path), vers}}
	x.Syntax.Stmt = append(x.Syntax.Stmt, syntax)

End:
	x.Require = append(x.Require, &Require{module.Version{Path: path, Version: vers}, syntax})
}

func (f *File) SetRequire(req []module.Version) {
	need := make(map[string]string)
	for _, m := range req {
		need[m.Path] = m.Version
	}

	for _, r := range f.Require {
		if v, ok := need[r.Mod.Path]; ok {
			r.Mod.Version = v
		}
	}

	var newStmts []Expr
	for _, stmt := range f.Syntax.Stmt {
		switch stmt := stmt.(type) {
		case *LineBlock:
			if len(stmt.Token) > 0 && stmt.Token[0] == "require" {
				var newLines []*Line
				for _, line := range stmt.Line {
					if p, err := strconv.Unquote(line.Token[0]); err == nil && need[p] != "" {
						line.Token[1] = need[p]
						delete(need, p)
						newLines = append(newLines, line)
					}
				}
				if len(newLines) == 0 {
					continue // drop stmt
				}
				stmt.Line = newLines
			}

		case *Line:
			if len(stmt.Token) > 0 && stmt.Token[0] == "require" {
				if p, err := strconv.Unquote(stmt.Token[1]); err == nil && need[p] != "" {
					stmt.Token[2] = need[p]
					delete(need, p)
				} else {
					continue // drop stmt
				}
			}
		}
		newStmts = append(newStmts, stmt)
	}
	f.Syntax.Stmt = newStmts

	for path, vers := range need {
		f.AddRequire(path, vers)
	}
	f.SortBlocks()
}

func (f *File) SortBlocks() {
	f.removeDups() // otherwise sorting is unsafe

	for _, stmt := range f.Syntax.Stmt {
		block, ok := stmt.(*LineBlock)
		if !ok {
			continue
		}
		sort.Slice(block.Line, func(i, j int) bool {
			li := block.Line[i]
			lj := block.Line[j]
			for k := 0; k < len(li.Token) && k < len(lj.Token); k++ {
				if li.Token[k] != lj.Token[k] {
					return li.Token[k] < lj.Token[k]
				}
			}
			return len(li.Token) < len(lj.Token)
		})
	}
}

func (f *File) removeDups() {
	have := make(map[module.Version]bool)
	kill := make(map[*Line]bool)
	for _, x := range f.Exclude {
		if have[x.Mod] {
			kill[x.Syntax] = true
			continue
		}
		have[x.Mod] = true
	}
	var excl []*Exclude
	for _, x := range f.Exclude {
		if !kill[x.Syntax] {
			excl = append(excl, x)
		}
	}
	f.Exclude = excl

	have = make(map[module.Version]bool)
	// Later replacements take priority over earlier ones.
	for i := len(f.Replace) - 1; i >= 0; i-- {
		x := f.Replace[i]
		if have[x.Old] {
			kill[x.Syntax] = true
			continue
		}
		have[x.Old] = true
	}
	var repl []*Replace
	for _, x := range f.Replace {
		if !kill[x.Syntax] {
			repl = append(repl, x)
		}
	}
	f.Replace = repl

	var stmts []Expr
	for _, stmt := range f.Syntax.Stmt {
		switch stmt := stmt.(type) {
		case *Line:
			if kill[stmt] {
				continue
			}
		case *LineBlock:
			var lines []*Line
			for _, line := range stmt.Line {
				if !kill[line] {
					lines = append(lines, line)
				}
			}
			stmt.Line = lines
			if len(lines) == 0 {
				continue
			}
		}
		stmts = append(stmts, stmt)
	}
	f.Syntax.Stmt = stmts
}
