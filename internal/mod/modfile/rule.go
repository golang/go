// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modfile implements a parser and formatter for go.mod files.
//
// The go.mod syntax is described in
// https://golang.org/cmd/go/#hdr-The_go_mod_file.
//
// The Parse and ParseLax functions both parse a go.mod file and return an
// abstract syntax tree. ParseLax ignores unknown statements and may be used to
// parse go.mod files that may have been developed with newer versions of Go.
//
// The File struct returned by Parse and ParseLax represent an abstract
// go.mod file. File has several methods like AddNewRequire and DropReplace
// that can be used to programmatically edit a file.
//
// The Format function formats a File back to a byte slice which can be
// written to a file.
package modfile

import (
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/mod/lazyregexp"
)

// A WorkFile is the parsed, interpreted form of a go.work file.
type WorkFile struct {
	Go        *modfile.Go
	Directory []*Directory
	Replace   []*modfile.Replace

	Syntax *modfile.FileSyntax
}

// A Directory is a single directory statement.
type Directory struct {
	DiskPath   string // TODO(matloob): Replace uses module.Version for new. Do that here?
	ModulePath string // Module path in the comment.
	Syntax     *modfile.Line
}

// Parse parses and returns a go.work file.
//
// file is the name of the file, used in positions and errors.
//
// data is the content of the file.
//
// fix is an optional function that canonicalizes module versions.
// If fix is nil, all module versions must be canonical (module.CanonicalVersion
// must return the same string).
func ParseWork(file string, data []byte, fix modfile.VersionFixer) (*WorkFile, error) {
	return parseToWorkFile(file, data, fix, true)
}

var GoVersionRE = lazyregexp.New(`^([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func parseToWorkFile(file string, data []byte, fix modfile.VersionFixer, strict bool) (parsed *WorkFile, err error) {
	fs, err := parse(file, data)
	if err != nil {
		return nil, err
	}
	f := &WorkFile{
		Syntax: fs,
	}
	var errs modfile.ErrorList

	for _, x := range fs.Stmt {
		switch x := x.(type) {
		case *modfile.Line:
			f.add(&errs, nil, x, x.Token[0], x.Token[1:], fix, strict)

		case *modfile.LineBlock:
			if len(x.Token) > 1 {
				if strict {
					errs = append(errs, modfile.Error{
						Filename: file,
						Pos:      x.Start,
						Err:      fmt.Errorf("unknown block type: %s", strings.Join(x.Token, " ")),
					})
				}
				continue
			}
			switch x.Token[0] {
			default:
				if strict {
					errs = append(errs, modfile.Error{
						Filename: file,
						Pos:      x.Start,
						Err:      fmt.Errorf("unknown block type: %s", strings.Join(x.Token, " ")),
					})
				}
				continue
			case "module", "directory", "replace":
				for _, l := range x.Line {
					f.add(&errs, x, l, x.Token[0], l.Token, fix, strict)
				}
			}
		}
	}

	if len(errs) > 0 {
		return nil, errs
	}
	return f, nil
}

func (f *WorkFile) add(errs *modfile.ErrorList, block *modfile.LineBlock, line *modfile.Line, verb string, args []string, fix modfile.VersionFixer, strict bool) {
	// If strict is false, this module is a dependency.
	// We ignore all unknown directives as well as main-module-only
	// directives like replace and exclude. It will work better for
	// forward compatibility if we can depend on modules that have unknown
	// statements (presumed relevant only when acting as the main module)
	// and simply ignore those statements.
	if !strict {
		switch verb {
		case "go", "module", "retract", "require":
			// want these even for dependency go.mods
		default:
			return
		}
	}

	wrapModPathError := func(modPath string, err error) {
		*errs = append(*errs, modfile.Error{
			Filename: f.Syntax.Name,
			Pos:      line.Start,
			ModPath:  modPath,
			Verb:     verb,
			Err:      err,
		})
	}
	wrapError := func(err error) {
		*errs = append(*errs, modfile.Error{
			Filename: f.Syntax.Name,
			Pos:      line.Start,
			Err:      err,
		})
	}
	errorf := func(format string, args ...interface{}) {
		wrapError(fmt.Errorf(format, args...))
	}

	switch verb {
	default:
		errorf("unknown directive: %s", verb)

	case "go":
		if f.Go != nil {
			errorf("repeated go statement")
			return
		}
		if len(args) != 1 {
			errorf("go directive expects exactly one argument")
			return
		} else if !GoVersionRE.MatchString(args[0]) {
			errorf("invalid go version '%s': must match format 1.23", args[0])
			return
		}

		f.Go = &modfile.Go{Syntax: line}
		f.Go.Version = args[0]

	case "directory":
		if len(args) != 1 {
			errorf("usage: %s ../local/directory", verb) // TODO(matloob) better example; most directories will be subdirectories of go.work dir
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			errorf("invalid quoted string: %v", err)
			return
		}
		f.Directory = append(f.Directory, &Directory{
			DiskPath: s,
			Syntax:   line,
		})

	case "replace":
		arrow := 2
		if len(args) >= 2 && args[1] == "=>" {
			arrow = 1
		}
		if len(args) < arrow+2 || len(args) > arrow+3 || args[arrow] != "=>" {
			errorf("usage: %s module/path [v1.2.3] => other/module v1.4\n\t or %s module/path [v1.2.3] => ../local/directory", verb, verb)
			return
		}
		s, err := parseString(&args[0])
		if err != nil {
			errorf("invalid quoted string: %v", err)
			return
		}
		pathMajor, err := modulePathMajor(s)
		if err != nil {
			wrapModPathError(s, err)
			return
		}
		var v string
		if arrow == 2 {
			v, err = parseVersion(verb, s, &args[1], fix)
			if err != nil {
				wrapError(err)
				return
			}
			if err := module.CheckPathMajor(v, pathMajor); err != nil {
				wrapModPathError(s, err)
				return
			}
		}
		ns, err := parseString(&args[arrow+1])
		if err != nil {
			errorf("invalid quoted string: %v", err)
			return
		}
		nv := ""
		if len(args) == arrow+2 {
			if !IsDirectoryPath(ns) {
				errorf("replacement module without version must be directory path (rooted or starting with ./ or ../)")
				return
			}
			if filepath.Separator == '/' && strings.Contains(ns, `\`) {
				errorf("replacement directory appears to be Windows path (on a non-windows system)")
				return
			}
		}
		if len(args) == arrow+3 {
			nv, err = parseVersion(verb, ns, &args[arrow+2], fix)
			if err != nil {
				wrapError(err)
				return
			}
			if IsDirectoryPath(ns) {
				errorf("replacement module directory path %q cannot have version", ns)
				return
			}
		}
		f.Replace = append(f.Replace, &modfile.Replace{
			Old:    module.Version{Path: s, Version: v},
			New:    module.Version{Path: ns, Version: nv},
			Syntax: line,
		})
	}
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
		switch r {
		case ' ', '"', '\'', '`':
			return true

		case '(', ')', '[', ']', '{', '}', ',':
			if len(s) > 1 {
				return true
			}

		default:
			if !unicode.IsPrint(r) {
				return true
			}
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

func parseVersion(verb string, path string, s *string, fix modfile.VersionFixer) (string, error) {
	t, err := parseString(s)
	if err != nil {
		return "", &modfile.Error{
			Verb:    verb,
			ModPath: path,
			Err: &module.InvalidVersionError{
				Version: *s,
				Err:     err,
			},
		}
	}
	if fix != nil {
		fixed, err := fix(path, t)
		if err != nil {
			if err, ok := err.(*module.ModuleError); ok {
				return "", &modfile.Error{
					Verb:    verb,
					ModPath: path,
					Err:     err.Err,
				}
			}
			return "", err
		}
		t = fixed
	} else {
		cv := module.CanonicalVersion(t)
		if cv == "" {
			return "", &modfile.Error{
				Verb:    verb,
				ModPath: path,
				Err: &module.InvalidVersionError{
					Version: t,
					Err:     errors.New("must be of the form v1.2.3"),
				},
			}
		}
		t = cv
	}
	*s = t
	return *s, nil
}

func modulePathMajor(path string) (string, error) {
	_, major, ok := module.SplitPathVersion(path)
	if !ok {
		return "", fmt.Errorf("invalid module path")
	}
	return major, nil
}
