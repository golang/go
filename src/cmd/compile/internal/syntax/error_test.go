// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a regression test harness for syntax errors.
// The files in the testdata directory are parsed and the reported
// errors are compared against the errors declared in those files.
//
// Errors are declared in place in the form of "error comments",
// just before (or on the same line as) the offending token.
//
// Error comments must be of the form // ERROR rx or /* ERROR rx */
// where rx is a regular expression that matches the reported error
// message. The rx text comprises the comment text after "ERROR ",
// with any white space around it stripped.
//
// If the line comment form is used, the reported error's line must
// match the line of the error comment.
//
// If the regular comment form is used, the reported error's position
// must match the position of the token immediately following the
// error comment. Thus, /* ERROR ... */ comments should appear
// immediately before the position where the error is reported.
//
// Currently, the test harness only supports one error comment per
// token. If multiple error comments appear before a token, only
// the last one is considered.

package syntax

import (
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

const testdata = "testdata" // directory containing test files

var print = flag.Bool("print", false, "only print errors")

// A position represents a source position in the current file.
type position struct {
	line, col uint
}

func (pos position) String() string {
	return fmt.Sprintf("%d:%d", pos.line, pos.col)
}

func sortedPositions(m map[position]string) []position {
	list := make([]position, len(m))
	i := 0
	for pos := range m {
		list[i] = pos
		i++
	}
	sort.Slice(list, func(i, j int) bool {
		a, b := list[i], list[j]
		return a.line < b.line || a.line == b.line && a.col < b.col
	})
	return list
}

// declaredErrors returns a map of source positions to error
// patterns, extracted from error comments in the given file.
// Error comments in the form of line comments use col = 0
// in their position.
func declaredErrors(t *testing.T, filename string) map[position]string {
	f, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	declared := make(map[position]string)

	var s scanner
	var pattern string
	s.init(f, func(line, col uint, msg string) {
		// errors never start with '/' so they are automatically excluded here
		switch {
		case strings.HasPrefix(msg, "// ERROR "):
			// we can't have another comment on the same line - just add it
			declared[position{s.line, 0}] = strings.TrimSpace(msg[9:])
		case strings.HasPrefix(msg, "/* ERROR "):
			// we may have more comments before the next token - collect them
			pattern = strings.TrimSpace(msg[9 : len(msg)-2])
		}
	}, comments)

	// consume file
	for {
		s.next()
		if pattern != "" {
			declared[position{s.line, s.col}] = pattern
			pattern = ""
		}
		if s.tok == _EOF {
			break
		}
	}

	return declared
}

func testSyntaxErrors(t *testing.T, filename string) {
	declared := declaredErrors(t, filename)
	if *print {
		fmt.Println("Declared errors:")
		for _, pos := range sortedPositions(declared) {
			fmt.Printf("%s:%s: %s\n", filename, pos, declared[pos])
		}

		fmt.Println()
		fmt.Println("Reported errors:")
	}

	f, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	ParseFile(filename, func(err error) {
		e, ok := err.(Error)
		if !ok {
			return
		}

		if *print {
			fmt.Println(err)
			return
		}

		orig := position{e.Pos.Line(), e.Pos.Col()}
		pos := orig
		pattern, found := declared[pos]
		if !found {
			// try line comment (only line must match)
			pos = position{e.Pos.Line(), 0}
			pattern, found = declared[pos]
		}
		if found {
			rx, err := regexp.Compile(pattern)
			if err != nil {
				t.Errorf("%s:%s: %v", filename, pos, err)
				return
			}
			if match := rx.MatchString(e.Msg); !match {
				t.Errorf("%s:%s: %q does not match %q", filename, pos, e.Msg, pattern)
				return
			}
			// we have a match - eliminate this error
			delete(declared, pos)
		} else {
			t.Errorf("%s:%s: unexpected error: %s", filename, orig, e.Msg)
		}
	}, nil, CheckBranches)

	if *print {
		fmt.Println()
		return // we're done
	}

	// report expected but not reported errors
	for pos, pattern := range declared {
		t.Errorf("%s:%s: missing error: %s", filename, pos, pattern)
	}
}

func TestSyntaxErrors(t *testing.T) {
	testenv.MustHaveGoBuild(t) // we need access to source (testdata)

	list, err := os.ReadDir(testdata)
	if err != nil {
		t.Fatal(err)
	}
	for _, fi := range list {
		name := fi.Name()
		if !fi.IsDir() && !strings.HasPrefix(name, ".") {
			testSyntaxErrors(t, filepath.Join(testdata, name))
		}
	}
}
