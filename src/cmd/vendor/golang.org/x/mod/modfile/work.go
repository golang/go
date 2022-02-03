// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"fmt"
	"sort"
	"strings"
)

// A WorkFile is the parsed, interpreted form of a go.work file.
type WorkFile struct {
	Go      *Go
	Use     []*Use
	Replace []*Replace

	Syntax *FileSyntax
}

// A Use is a single directory statement.
type Use struct {
	Path       string // Use path of module.
	ModulePath string // Module path in the comment.
	Syntax     *Line
}

// ParseWork parses and returns a go.work file.
//
// file is the name of the file, used in positions and errors.
//
// data is the content of the file.
//
// fix is an optional function that canonicalizes module versions.
// If fix is nil, all module versions must be canonical (module.CanonicalVersion
// must return the same string).
func ParseWork(file string, data []byte, fix VersionFixer) (*WorkFile, error) {
	fs, err := parse(file, data)
	if err != nil {
		return nil, err
	}
	f := &WorkFile{
		Syntax: fs,
	}
	var errs ErrorList

	for _, x := range fs.Stmt {
		switch x := x.(type) {
		case *Line:
			f.add(&errs, x, x.Token[0], x.Token[1:], fix)

		case *LineBlock:
			if len(x.Token) > 1 {
				errs = append(errs, Error{
					Filename: file,
					Pos:      x.Start,
					Err:      fmt.Errorf("unknown block type: %s", strings.Join(x.Token, " ")),
				})
				continue
			}
			switch x.Token[0] {
			default:
				errs = append(errs, Error{
					Filename: file,
					Pos:      x.Start,
					Err:      fmt.Errorf("unknown block type: %s", strings.Join(x.Token, " ")),
				})
				continue
			case "use", "replace":
				for _, l := range x.Line {
					f.add(&errs, l, x.Token[0], l.Token, fix)
				}
			}
		}
	}

	if len(errs) > 0 {
		return nil, errs
	}
	return f, nil
}

// Cleanup cleans up the file f after any edit operations.
// To avoid quadratic behavior, modifications like DropRequire
// clear the entry but do not remove it from the slice.
// Cleanup cleans out all the cleared entries.
func (f *WorkFile) Cleanup() {
	w := 0
	for _, r := range f.Use {
		if r.Path != "" {
			f.Use[w] = r
			w++
		}
	}
	f.Use = f.Use[:w]

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

func (f *WorkFile) AddGoStmt(version string) error {
	if !GoVersionRE.MatchString(version) {
		return fmt.Errorf("invalid language version string %q", version)
	}
	if f.Go == nil {
		stmt := &Line{Token: []string{"go", version}}
		f.Go = &Go{
			Version: version,
			Syntax:  stmt,
		}
		// Find the first non-comment-only block that's and add
		// the go statement before it. That will keep file comments at the top.
		i := 0
		for i = 0; i < len(f.Syntax.Stmt); i++ {
			if _, ok := f.Syntax.Stmt[i].(*CommentBlock); !ok {
				break
			}
		}
		f.Syntax.Stmt = append(append(f.Syntax.Stmt[:i:i], stmt), f.Syntax.Stmt[i:]...)
	} else {
		f.Go.Version = version
		f.Syntax.updateLine(f.Go.Syntax, "go", version)
	}
	return nil
}

func (f *WorkFile) AddUse(diskPath, modulePath string) error {
	need := true
	for _, d := range f.Use {
		if d.Path == diskPath {
			if need {
				d.ModulePath = modulePath
				f.Syntax.updateLine(d.Syntax, "use", AutoQuote(diskPath))
				need = false
			} else {
				d.Syntax.markRemoved()
				*d = Use{}
			}
		}
	}

	if need {
		f.AddNewUse(diskPath, modulePath)
	}
	return nil
}

func (f *WorkFile) AddNewUse(diskPath, modulePath string) {
	line := f.Syntax.addLine(nil, "use", AutoQuote(diskPath))
	f.Use = append(f.Use, &Use{Path: diskPath, ModulePath: modulePath, Syntax: line})
}

func (f *WorkFile) SetUse(dirs []*Use) {
	need := make(map[string]string)
	for _, d := range dirs {
		need[d.Path] = d.ModulePath
	}

	for _, d := range f.Use {
		if modulePath, ok := need[d.Path]; ok {
			d.ModulePath = modulePath
		} else {
			d.Syntax.markRemoved()
			*d = Use{}
		}
	}

	// TODO(#45713): Add module path to comment.

	for diskPath, modulePath := range need {
		f.AddNewUse(diskPath, modulePath)
	}
	f.SortBlocks()
}

func (f *WorkFile) DropUse(path string) error {
	for _, d := range f.Use {
		if d.Path == path {
			d.Syntax.markRemoved()
			*d = Use{}
		}
	}
	return nil
}

func (f *WorkFile) AddReplace(oldPath, oldVers, newPath, newVers string) error {
	return addReplace(f.Syntax, &f.Replace, oldPath, oldVers, newPath, newVers)
}

func (f *WorkFile) DropReplace(oldPath, oldVers string) error {
	for _, r := range f.Replace {
		if r.Old.Path == oldPath && r.Old.Version == oldVers {
			r.Syntax.markRemoved()
			*r = Replace{}
		}
	}
	return nil
}

func (f *WorkFile) SortBlocks() {
	f.removeDups() // otherwise sorting is unsafe

	for _, stmt := range f.Syntax.Stmt {
		block, ok := stmt.(*LineBlock)
		if !ok {
			continue
		}
		sort.SliceStable(block.Line, func(i, j int) bool {
			return lineLess(block.Line[i], block.Line[j])
		})
	}
}

// removeDups removes duplicate replace directives.
//
// Later replace directives take priority.
//
// require directives are not de-duplicated. That's left up to higher-level
// logic (MVS).
//
// retract directives are not de-duplicated since comments are
// meaningful, and versions may be retracted multiple times.
func (f *WorkFile) removeDups() {
	removeDups(f.Syntax, nil, &f.Replace)
}
