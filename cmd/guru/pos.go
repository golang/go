// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file defines utilities for working with file positions.

import (
	"fmt"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/buildutil"
)

// parseOctothorpDecimal returns the numeric value if s matches "#%d",
// otherwise -1.
func parseOctothorpDecimal(s string) int {
	if s != "" && s[0] == '#' {
		if s, err := strconv.ParseInt(s[1:], 10, 32); err == nil {
			return int(s)
		}
	}
	return -1
}

// parsePos parses a string of the form "file:pos" or
// file:start,end" where pos, start, end match #%d and represent byte
// offsets, and returns its components.
//
// (Numbers without a '#' prefix are reserved for future use,
// e.g. to indicate line/column positions.)
func parsePos(pos string) (filename string, startOffset, endOffset int, err error) {
	if pos == "" {
		err = fmt.Errorf("no source position specified")
		return
	}

	colon := strings.LastIndex(pos, ":")
	if colon < 0 {
		err = fmt.Errorf("bad position syntax %q", pos)
		return
	}
	filename, offset := pos[:colon], pos[colon+1:]
	startOffset = -1
	endOffset = -1
	if comma := strings.Index(offset, ","); comma < 0 {
		// e.g. "foo.go:#123"
		startOffset = parseOctothorpDecimal(offset)
		endOffset = startOffset
	} else {
		// e.g. "foo.go:#123,#456"
		startOffset = parseOctothorpDecimal(offset[:comma])
		endOffset = parseOctothorpDecimal(offset[comma+1:])
	}
	if startOffset < 0 || endOffset < 0 {
		err = fmt.Errorf("invalid offset %q in query position", offset)
		return
	}
	return
}

// fileOffsetToPos translates the specified file-relative byte offsets
// into token.Pos form.  It returns an error if the file was not found
// or the offsets were out of bounds.
func fileOffsetToPos(file *token.File, startOffset, endOffset int) (start, end token.Pos, err error) {
	// Range check [start..end], inclusive of both end-points.

	if 0 <= startOffset && startOffset <= file.Size() {
		start = file.Pos(int(startOffset))
	} else {
		err = fmt.Errorf("start position is beyond end of file")
		return
	}

	if 0 <= endOffset && endOffset <= file.Size() {
		end = file.Pos(int(endOffset))
	} else {
		err = fmt.Errorf("end position is beyond end of file")
		return
	}

	return
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
func sameFile(x, y string) bool {
	if filepath.Base(x) == filepath.Base(y) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}

// fastQueryPos parses the position string and returns a queryPos.
// It parses only a single file and does not run the type checker.
func fastQueryPos(ctxt *build.Context, pos string) (*queryPos, error) {
	filename, startOffset, endOffset, err := parsePos(pos)
	if err != nil {
		return nil, err
	}

	// Parse the file, opening it the file via the build.Context
	// so that we observe the effects of the -modified flag.
	fset := token.NewFileSet()
	cwd, _ := os.Getwd()
	f, err := buildutil.ParseFile(fset, ctxt, nil, cwd, filename, parser.Mode(0))
	// ParseFile usually returns a partial file along with an error.
	// Only fail if there is no file.
	if f == nil {
		return nil, err
	}
	if !f.Pos().IsValid() {
		return nil, fmt.Errorf("%s is not a Go source file", filename)
	}

	start, end, err := fileOffsetToPos(fset.File(f.Pos()), startOffset, endOffset)
	if err != nil {
		return nil, err
	}

	path, exact := astutil.PathEnclosingInterval(f, start, end)
	if path == nil {
		return nil, fmt.Errorf("no syntax here")
	}

	return &queryPos{fset, start, end, path, exact, nil}, nil
}
