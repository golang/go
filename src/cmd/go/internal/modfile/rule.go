// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"bytes"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

// A File is the parsed, interpreted form of a go.mod file.
type File struct {
	Module  *Module
	Go      *Go
	Require []*Require
	Exclude []*Exclude
	Replace []*Replace

	Syntax *FileSyntax
}

// A Module is the module statement.
type Module struct {
	Mod    module.Version
	Syntax *Line
}

// A Go is the go statement.
type Go struct {
	Version string // "1.23"
	Syntax  *Line
}

// A Require is a single require statement.
type Require struct {
	Mod      module.Version
	Indirect bool // has "// indirect" comment
	Syntax   *Line
}

// An Exclude is a single exclude statement.
type Exclude struct {
	Mod    module.Version
	Syntax *Line
}

// A Replace is a single replace statement.
type Replace struct {
	Old    module.Version
	New    module.Version
	Syntax *Line
}

func (f *File) AddModuleStmt(path string) error {
	if f.Syntax == nil {
		f.Syntax = new(FileSyntax)
	}
	if f.Module == nil {
		f.Module = &Module{
			Mod:    module.Version{Path: path},
			Syntax: f.Syntax.addLine(nil, "module", AutoQuote(path)),
		}
	} else {
		f.Module.Mod.Path = path
		f.Syntax.updateLine(f.Module.Syntax, "module", AutoQuote(path))
	}
	return nil
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

// Parse parses the data, reported in errors as being from file,
// into a File struct. It applies fix, if non-nil, to canonicalize all module versions found.
func Parse(file string, data []byte, fix VersionFixer) (*File, error) {
	return parseToFile(file, data, fix, true)
}

// ParseLax is like Parse but ignores unknown statements.
// It is used when parsing go.mod files other than the main module,
// under the theory that most statement types we add in the future will
// only apply in the main module, like exclude and replace,
// and so we get better gradual deployments if old go commands
// simply ignore those statements when found in go.mod files
// in dependencies.
func ParseLax(file string, data []byte, fix VersionFixer) (*File, error) {
	return parseToFile(file, data, fix, false)
}

func parseToFile(file string, data []byte, fix VersionFixer, strict bool) (*File, error) {
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
			f.add(&errs, x, x.Token[0], x.Token[1:], fix, strict)

		case *LineBlock:
			if len(x.Token) > 1 {
				if strict {
					fmt.Fprintf(&errs, "%s:%d: unknown block type: %s\n", file, x.Start.Line, strings.Join(x.Token, " "))
				}
				continue
			}
			switch x.Token[0] {
			default:
				if strict {
					fmt.Fprintf(&errs, "%s:%d: unknown block type: %s\n", file, x.Start.Line, strings.Join(x.Token, " "))
				}
				continue
			case "module", "require", "exclude", "replace":
				for _, l := range x.Line {
					f.add(&errs, l, x.Token[0], l.Token, fix, strict)
				}
			}
		}
	}

	if errs.Len() > 0 {
		return nil, errors.New(strings.TrimRight(errs.String(), "\n"))
	}
	return f, nil
}

var goVersionRE = regexp.MustCompile(`([1-9][0-9]*)\.(0|[1-9][0-9]*)`)

func (f *File) add(errs *bytes.Buffer, line *Line, verb string, args []string, fix VersionFixer, strict bool) {
	// If strict is false, this module is a dependency.
	// We ignore all unknown directives as well as main-module-only
	// directives like replace and exclude. It will work better for
	// forward compatibility if we can depend on modules that have unknown
	// statements (presumed relevant only when acting as the main module)
	// and simply ignore those statements.
	if !strict {
		switch verb {
		case "module", "require", "go":
			// want these even for dependency go.mods
		default:
			return
		}
	}

	switch verb {
	default:
		fmt.Fprintf(errs, "%s:%d: unknown directive: %s\n", f.Syntax.Name, line.Start.Line, verb)

	case "go":
		if f.Go != nil {
			fmt.Fprintf(errs, "%s:%d: repeated go statement\n", f.Syntax.Name, line.Start.Line)
			return
		}
		if len(args) != 1 || !goVersionRE.MatchString(args[0]) {
			fmt.Fprintf(errs, "%s:%d: usage: go 1.23\n", f.Syntax.Name, line.Start.Line)
			return
		}
		f.Go = &Go{Syntax: line}
		f.Go.Version = args[0]
	case "module":
		if f.Module != nil {
			fmt.Fprintf(errs, "%s:%d: repeated module statement\n", f.Syntax.Name, line.Start.Line)
			return
		}
		f.Module = &Module{Syntax: line}
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
		pathMajor, err := modulePathMajor(s)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		if !module.MatchPathMajor(v, pathMajor) {
			if pathMajor == "" {
				pathMajor = "v0 or v1"
			}
			fmt.Fprintf(errs, "%s:%d: invalid module: %s should be %s, not %s (%s)\n", f.Syntax.Name, line.Start.Line, s, pathMajor, semver.Major(v), v)
			return
		}
		if verb == "require" {
			f.Require = append(f.Require, &Require{
				Mod:      module.Version{Path: s, Version: v},
				Syntax:   line,
				Indirect: isIndirect(line),
			})
		} else {
			f.Exclude = append(f.Exclude, &Exclude{
				Mod:    module.Version{Path: s, Version: v},
				Syntax: line,
			})
		}
	case "replace":
		arrow := 2
		if len(args) >= 2 && args[1] == "=>" {
			arrow = 1
		}
		if len(args) < arrow+2 || len(args) > arrow+3 || args[arrow] != "=>" {
			fmt.Fprintf(errs, "%s:%d: usage: %s module/path [v1.2.3] => other/module v1.4\n\t or %s module/path [v1.2.3] => ../local/directory\n", f.Syntax.Name, line.Start.Line, verb, verb)
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		pathMajor, err := modulePathMajor(s)
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		var v string
		if arrow == 2 {
			old := args[1]
			v, err = parseVersion(s, &args[1], fix)
			if err != nil {
				fmt.Fprintf(errs, "%s:%d: invalid module version %v: %v\n", f.Syntax.Name, line.Start.Line, old, err)
				return
			}
			if !module.MatchPathMajor(v, pathMajor) {
				if pathMajor == "" {
					pathMajor = "v0 or v1"
				}
				fmt.Fprintf(errs, "%s:%d: invalid module: %s should be %s, not %s (%s)\n", f.Syntax.Name, line.Start.Line, s, pathMajor, semver.Major(v), v)
				return
			}
		}
		ns, err := parseString(&args[arrow+1])
		if err != nil {
			fmt.Fprintf(errs, "%s:%d: invalid quoted string: %v\n", f.Syntax.Name, line.Start.Line, err)
			return
		}
		nv := ""
		if len(args) == arrow+2 {
			if !IsDirectoryPath(ns) {
				fmt.Fprintf(errs, "%s:%d: replacement module without version must be directory path (rooted or starting with ./ or ../)\n", f.Syntax.Name, line.Start.Line)
				return
			}
			if filepath.Separator == '/' && strings.Contains(ns, `\`) {
				fmt.Fprintf(errs, "%s:%d: replacement directory appears to be Windows path (on a non-windows system)\n", f.Syntax.Name, line.Start.Line)
				return
			}
		}
		if len(args) == arrow+3 {
			old := args[arrow+1]
			nv, err = parseVersion(ns, &args[arrow+2], fix)
			if err != nil {
				fmt.Fprintf(errs, "%s:%d: invalid module version %v: %v\n", f.Syntax.Name, line.Start.Line, old, err)
				return
			}
			if IsDirectoryPath(ns) {
				fmt.Fprintf(errs, "%s:%d: replacement module directory path %q cannot have version\n", f.Syntax.Name, line.Start.Line, ns)
				return
			}
		}
		f.Replace = append(f.Replace, &Replace{
			Old:    module.Version{Path: s, Version: v},
			New:    module.Version{Path: ns, Version: nv},
			Syntax: line,
		})
	}
}

// isIndirect reports whether line has a "// indirect" comment,
// meaning it is in go.mod only for its effect on indirect dependencies,
// so that it can be dropped entirely once the effective version of the
// indirect dependency reaches the given minimum version.
func isIndirect(line *Line) bool {
	if len(line.Suffix) == 0 {
		return false
	}
	f := strings.Fields(line.Suffix[0].Token)
	return (len(f) == 2 && f[1] == "indirect" || len(f) > 2 && f[1] == "indirect;") && f[0] == "//"
}

// setIndirect sets line to have (or not have) a "// indirect" comment.
func setIndirect(line *Line, indirect bool) {
	if isIndirect(line) == indirect {
		return
	}
	if indirect {
		// Adding comment.
		if len(line.Suffix) == 0 {
			// New comment.
			line.Suffix = []Comment{{Token: "// indirect", Suffix: true}}
			return
		}
		// Insert at beginning of existing comment.
		com := &line.Suffix[0]
		space := " "
		if len(com.Token) > 2 && com.Token[2] == ' ' || com.Token[2] == '\t' {
			space = ""
		}
		com.Token = "// indirect;" + space + com.Token[2:]
		return
	}

	// Removing comment.
	f := strings.Fields(line.Suffix[0].Token)
	if len(f) == 2 {
		// Remove whole comment.
		line.Suffix = nil
		return
	}

	// Remove comment prefix.
	com := &line.Suffix[0]
	i := strings.Index(com.Token, "indirect;")
	com.Token = "//" + com.Token[i+len("indirect;"):]
}

// IsDirectoryPath reports whether the given path should be interpreted
// as a directory path. Just like on the go command line, relative paths
// and rooted paths are directory paths; the rest are module paths.
func IsDirectoryPath(ns string) bool {
	// Because go.mod files can move from one system to another,
	// we check all known path syntaxes, both Unix and Windows.
	return strings.HasPrefix(ns, "./") || strings.HasPrefix(ns, "../") || strings.HasPrefix(ns, "/") ||
		strings.HasPrefix(ns, `.\`) || strings.HasPrefix(ns, `..\`) || strings.HasPrefix(ns, `\`) ||
		len(ns) >= 2 && ('A' <= ns[0] && ns[0] <= 'Z' || 'a' <= ns[0] && ns[0] <= 'z') && ns[1] == ':'
}

// MustQuote reports whether s must be quoted in order to appear as
// a single token in a go.mod line.
func MustQuote(s string) bool {
	for _, r := range s {
		if !unicode.IsPrint(r) || r == ' ' || r == '"' || r == '\'' || r == '`' {
			return true
		}
	}
	return s == "" || strings.Contains(s, "//") || strings.Contains(s, "/*")
}

// AutoQuote returns s or, if quoting is required for s to appear in a go.mod,
// the quotation of s.
func AutoQuote(s string) string {
	if MustQuote(s) {
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
	if v := module.CanonicalVersion(t); v != "" {
		*s = v
		return *s, nil
	}
	return "", fmt.Errorf("version must be of the form v1.2.3")
}

func modulePathMajor(path string) (string, error) {
	_, major, ok := module.SplitPathVersion(path)
	if !ok {
		return "", fmt.Errorf("invalid module path")
	}
	return major, nil
}

func (f *File) Format() ([]byte, error) {
	return Format(f.Syntax), nil
}

// Cleanup cleans up the file f after any edit operations.
// To avoid quadratic behavior, modifications like DropRequire
// clear the entry but do not remove it from the slice.
// Cleanup cleans out all the cleared entries.
func (f *File) Cleanup() {
	w := 0
	for _, r := range f.Require {
		if r.Mod.Path != "" {
			f.Require[w] = r
			w++
		}
	}
	f.Require = f.Require[:w]

	w = 0
	for _, x := range f.Exclude {
		if x.Mod.Path != "" {
			f.Exclude[w] = x
			w++
		}
	}
	f.Exclude = f.Exclude[:w]

	w = 0
	for _, r := range f.Replace {
		if r.Old.Path != "" {
			f.Replace[w] = r
			w++
		}
	}
	f.Replace = f.Replace[:w]

	f.Syntax.Cleanup()
}

func (f *File) AddRequire(path, vers string) error {
	need := true
	for _, r := range f.Require {
		if r.Mod.Path == path {
			if need {
				r.Mod.Version = vers
				f.Syntax.updateLine(r.Syntax, "require", AutoQuote(path), vers)
				need = false
			} else {
				f.Syntax.removeLine(r.Syntax)
				*r = Require{}
			}
		}
	}

	if need {
		f.AddNewRequire(path, vers, false)
	}
	return nil
}

func (f *File) AddNewRequire(path, vers string, indirect bool) {
	line := f.Syntax.addLine(nil, "require", AutoQuote(path), vers)
	setIndirect(line, indirect)
	f.Require = append(f.Require, &Require{module.Version{Path: path, Version: vers}, indirect, line})
}

func (f *File) SetRequire(req []*Require) {
	need := make(map[string]string)
	indirect := make(map[string]bool)
	for _, r := range req {
		need[r.Mod.Path] = r.Mod.Version
		indirect[r.Mod.Path] = r.Indirect
	}

	for _, r := range f.Require {
		if v, ok := need[r.Mod.Path]; ok {
			r.Mod.Version = v
			r.Indirect = indirect[r.Mod.Path]
		}
	}

	var newStmts []Expr
	for _, stmt := range f.Syntax.Stmt {
		switch stmt := stmt.(type) {
		case *LineBlock:
			if len(stmt.Token) > 0 && stmt.Token[0] == "require" {
				var newLines []*Line
				for _, line := range stmt.Line {
					if p, err := parseString(&line.Token[0]); err == nil && need[p] != "" {
						line.Token[1] = need[p]
						delete(need, p)
						setIndirect(line, indirect[p])
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
				if p, err := parseString(&stmt.Token[1]); err == nil && need[p] != "" {
					stmt.Token[2] = need[p]
					delete(need, p)
					setIndirect(stmt, indirect[p])
				} else {
					continue // drop stmt
				}
			}
		}
		newStmts = append(newStmts, stmt)
	}
	f.Syntax.Stmt = newStmts

	for path, vers := range need {
		f.AddNewRequire(path, vers, indirect[path])
	}
	f.SortBlocks()
}

func (f *File) DropRequire(path string) error {
	for _, r := range f.Require {
		if r.Mod.Path == path {
			f.Syntax.removeLine(r.Syntax)
			*r = Require{}
		}
	}
	return nil
}

func (f *File) AddExclude(path, vers string) error {
	var hint *Line
	for _, x := range f.Exclude {
		if x.Mod.Path == path && x.Mod.Version == vers {
			return nil
		}
		if x.Mod.Path == path {
			hint = x.Syntax
		}
	}

	f.Exclude = append(f.Exclude, &Exclude{Mod: module.Version{Path: path, Version: vers}, Syntax: f.Syntax.addLine(hint, "exclude", AutoQuote(path), vers)})
	return nil
}

func (f *File) DropExclude(path, vers string) error {
	for _, x := range f.Exclude {
		if x.Mod.Path == path && x.Mod.Version == vers {
			f.Syntax.removeLine(x.Syntax)
			*x = Exclude{}
		}
	}
	return nil
}

func (f *File) AddReplace(oldPath, oldVers, newPath, newVers string) error {
	need := true
	old := module.Version{Path: oldPath, Version: oldVers}
	new := module.Version{Path: newPath, Version: newVers}
	tokens := []string{"replace", AutoQuote(oldPath)}
	if oldVers != "" {
		tokens = append(tokens, oldVers)
	}
	tokens = append(tokens, "=>", AutoQuote(newPath))
	if newVers != "" {
		tokens = append(tokens, newVers)
	}

	var hint *Line
	for _, r := range f.Replace {
		if r.Old.Path == oldPath && (oldVers == "" || r.Old.Version == oldVers) {
			if need {
				// Found replacement for old; update to use new.
				r.New = new
				f.Syntax.updateLine(r.Syntax, tokens...)
				need = false
				continue
			}
			// Already added; delete other replacements for same.
			f.Syntax.removeLine(r.Syntax)
			*r = Replace{}
		}
		if r.Old.Path == oldPath {
			hint = r.Syntax
		}
	}
	if need {
		f.Replace = append(f.Replace, &Replace{Old: old, New: new, Syntax: f.Syntax.addLine(hint, tokens...)})
	}
	return nil
}

func (f *File) DropReplace(oldPath, oldVers string) error {
	for _, r := range f.Replace {
		if r.Old.Path == oldPath && r.Old.Version == oldVers {
			f.Syntax.removeLine(r.Syntax)
			*r = Replace{}
		}
	}
	return nil
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
