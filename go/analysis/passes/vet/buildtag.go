// +build ignore

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"unicode"
)

var (
	nl         = []byte("\n")
	slashSlash = []byte("//")
	plusBuild  = []byte("+build")
)

func badfLine(f *File, line int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(os.Stderr, "%s:%d: %s\n", f.name, line, msg)
	setExit(1)
}

// checkBuildTag checks that build tags are in the correct location and well-formed.
func checkBuildTag(f *File) {
	if !vet("buildtags") {
		return
	}

	// we must look at the raw lines, as build tags may appear in non-Go
	// files such as assembly files.
	lines := bytes.SplitAfter(f.content, nl)

	// lineWithComment reports whether a line corresponds to a comment in
	// the source file. If the source file wasn't Go, the function always
	// returns true.
	lineWithComment := func(line int) bool {
		if f.file == nil {
			// Current source file is not Go, so be conservative.
			return true
		}
		for _, group := range f.file.Comments {
			startLine := f.fset.Position(group.Pos()).Line
			endLine := f.fset.Position(group.End()).Line
			if startLine <= line && line <= endLine {
				return true
			}
		}
		return false
	}

	// Determine cutpoint where +build comments are no longer valid.
	// They are valid in leading // comments in the file followed by
	// a blank line.
	var cutoff int
	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			cutoff = i
			continue
		}
		if bytes.HasPrefix(line, slashSlash) {
			continue
		}
		break
	}

	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashSlash) {
			continue
		}
		if !bytes.Contains(line, plusBuild) {
			// Check that the comment contains "+build" early, to
			// avoid unnecessary lineWithComment calls that may
			// incur linear searches.
			continue
		}
		if !lineWithComment(i + 1) {
			// This is a line in a Go source file that looks like a
			// comment, but actually isn't - such as part of a raw
			// string.
			continue
		}

		text := bytes.TrimSpace(line[2:])
		if bytes.HasPrefix(text, plusBuild) {
			fields := bytes.Fields(text)
			if !bytes.Equal(fields[0], plusBuild) {
				// Comment is something like +buildasdf not +build.
				badfLine(f, i+1, "possible malformed +build comment")
				continue
			}
			if i >= cutoff {
				badfLine(f, i+1, "+build comment must appear before package clause and be followed by a blank line")
				continue
			}
			// Check arguments.
		Args:
			for _, arg := range fields[1:] {
				for _, elem := range strings.Split(string(arg), ",") {
					if strings.HasPrefix(elem, "!!") {
						badfLine(f, i+1, "invalid double negative in build constraint: %s", arg)
						break Args
					}
					elem = strings.TrimPrefix(elem, "!")
					for _, c := range elem {
						if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
							badfLine(f, i+1, "invalid non-alphanumeric build constraint: %s", arg)
							break Args
						}
					}
				}
			}
			continue
		}
		// Comment with +build but not at beginning.
		if i < cutoff {
			badfLine(f, i+1, "possible malformed +build comment")
			continue
		}
	}
}
