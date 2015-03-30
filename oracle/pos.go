package oracle

// This file defines utilities for working with file positions.

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
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

// parsePosFlag parses a string of the form "file:pos" or
// file:start,end" where pos, start, end match #%d and represent byte
// offsets, and returns its components.
//
// (Numbers without a '#' prefix are reserved for future use,
// e.g. to indicate line/column positions.)
//
func parsePosFlag(posFlag string) (filename string, startOffset, endOffset int, err error) {
	if posFlag == "" {
		err = fmt.Errorf("no source position specified (-pos flag)")
		return
	}

	colon := strings.LastIndex(posFlag, ":")
	if colon < 0 {
		err = fmt.Errorf("invalid source position -pos=%q", posFlag)
		return
	}
	filename, offset := posFlag[:colon], posFlag[colon+1:]
	startOffset = -1
	endOffset = -1
	if hyphen := strings.Index(offset, ","); hyphen < 0 {
		// e.g. "foo.go:#123"
		startOffset = parseOctothorpDecimal(offset)
		endOffset = startOffset
	} else {
		// e.g. "foo.go:#123,#456"
		startOffset = parseOctothorpDecimal(offset[:hyphen])
		endOffset = parseOctothorpDecimal(offset[hyphen+1:])
	}
	if startOffset < 0 || endOffset < 0 {
		err = fmt.Errorf("invalid -pos offset %q", offset)
		return
	}
	return
}

// findQueryPos searches fset for filename and translates the
// specified file-relative byte offsets into token.Pos form.  It
// returns an error if the file was not found or the offsets were out
// of bounds.
//
func findQueryPos(fset *token.FileSet, filename string, startOffset, endOffset int) (start, end token.Pos, err error) {
	var file *token.File
	fset.Iterate(func(f *token.File) bool {
		if sameFile(filename, f.Name()) {
			// (f.Name() is absolute)
			file = f
			return false // done
		}
		return true // continue
	})
	if file == nil {
		err = fmt.Errorf("couldn't find file containing position")
		return
	}

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
//
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

// fastQueryPos parses the -pos flag and returns a QueryPos.
// It parses only a single file, and does not run the type checker.
func fastQueryPos(posFlag string) (*queryPos, error) {
	filename, startOffset, endOffset, err := parsePosFlag(posFlag)
	if err != nil {
		return nil, err
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, nil, 0)
	if err != nil {
		return nil, err
	}

	start, end, err := findQueryPos(fset, filename, startOffset, endOffset)
	if err != nil {
		return nil, err
	}

	path, exact := astutil.PathEnclosingInterval(f, start, end)
	if path == nil {
		return nil, fmt.Errorf("no syntax here")
	}

	return &queryPos{fset, start, end, path, exact, nil}, nil
}
