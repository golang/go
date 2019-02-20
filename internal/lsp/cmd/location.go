// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"fmt"
	"go/token"
	"path/filepath"
	"regexp"
	"strconv"

	"golang.org/x/tools/internal/lsp/source"
)

type Location struct {
	Filename string   `json:"file"`
	Start    Position `json:"start"`
	End      Position `json:"end"`
}

type Position struct {
	Line   int `json:"line"`
	Column int `json:"column"`
	Offset int `json:"offset"`
}

func newLocation(fset *token.FileSet, r source.Range) Location {
	start := fset.Position(r.Start)
	end := fset.Position(r.End)
	// it should not be possible the following line to fail
	filename, _ := source.ToURI(start.Filename).Filename()
	return Location{
		Filename: filename,
		Start: Position{
			Line:   start.Line,
			Column: start.Column,
			Offset: fset.File(r.Start).Offset(r.Start),
		},
		End: Position{
			Line:   end.Line,
			Column: end.Column,
			Offset: fset.File(r.End).Offset(r.End),
		},
	}
}

var posRe = regexp.MustCompile(
	`(?P<file>.*):(?P<start>(?P<sline>\d+):(?P<scol>\d)+|#(?P<soff>\d+))(?P<end>:(?P<eline>\d+):(?P<ecol>\d+)|#(?P<eoff>\d+))?$`)

const (
	posReAll = iota
	posReFile
	posReStart
	posReSLine
	posReSCol
	posReSOff
	posReEnd
	posReELine
	posReECol
	posReEOff
)

func init() {
	names := posRe.SubexpNames()
	// verify all our submatch offsets are correct
	for name, index := range map[string]int{
		"file":  posReFile,
		"start": posReStart,
		"sline": posReSLine,
		"scol":  posReSCol,
		"soff":  posReSOff,
		"end":   posReEnd,
		"eline": posReELine,
		"ecol":  posReECol,
		"eoff":  posReEOff,
	} {
		if names[index] == name {
			continue
		}
		// try to find it
		for test := range names {
			if names[test] == name {
				panic(fmt.Errorf("Index for %s incorrect, wanted %v have %v", name, index, test))
			}
		}
		panic(fmt.Errorf("Subexp %s does not exist", name))
	}
}

// parseLocation parses a string of the form "file:pos" or
// file:start,end" where pos, start, end match either a byte offset in the
// form #%d or a line and column in the form %d,%d.
func parseLocation(value string) (Location, error) {
	var loc Location
	m := posRe.FindStringSubmatch(value)
	if m == nil {
		return loc, fmt.Errorf("bad location syntax %q", value)
	}
	loc.Filename = m[posReFile]
	if !filepath.IsAbs(loc.Filename) {
		loc.Filename, _ = filepath.Abs(loc.Filename) // ignore error
	}
	if m[posReSLine] != "" {
		v, err := strconv.ParseInt(m[posReSLine], 10, 32)
		if err != nil {
			return loc, err
		}
		loc.Start.Line = int(v)
		v, err = strconv.ParseInt(m[posReSCol], 10, 32)
		if err != nil {
			return loc, err
		}
		loc.Start.Column = int(v)
	} else {
		v, err := strconv.ParseInt(m[posReSOff], 10, 32)
		if err != nil {
			return loc, err
		}
		loc.Start.Offset = int(v)
	}
	if m[posReEnd] == "" {
		loc.End = loc.Start
	} else {
		if m[posReELine] != "" {
			v, err := strconv.ParseInt(m[posReELine], 10, 32)
			if err != nil {
				return loc, err
			}
			loc.End.Line = int(v)
			v, err = strconv.ParseInt(m[posReECol], 10, 32)
			if err != nil {
				return loc, err
			}
			loc.End.Column = int(v)
		} else {
			v, err := strconv.ParseInt(m[posReEOff], 10, 32)
			if err != nil {
				return loc, err
			}
			loc.End.Offset = int(v)
		}
	}
	return loc, nil
}

func (l Location) Format(f fmt.State, c rune) {
	// we should always have a filename
	fmt.Fprint(f, l.Filename)
	// are we in line:column format or #offset format
	fmt.Fprintf(f, ":%v", l.Start)
	if l.End != l.Start {
		fmt.Fprintf(f, ",%v", l.End)
	}
}

func (p Position) Format(f fmt.State, c rune) {
	// are we in line:column format or #offset format
	if p.Line > 0 {
		fmt.Fprintf(f, "%d:%d", p.Line, p.Column)
		return
	}
	fmt.Fprintf(f, "#%d", p.Offset)
}
