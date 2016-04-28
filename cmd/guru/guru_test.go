// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

// This file defines a test framework for guru queries.
//
// The files beneath testdata/src/main contain Go programs containing
// query annotations of the form:
//
//   @verb id "select"
//
// where verb is the query mode (e.g. "callers"), id is a unique name
// for this query, and "select" is a regular expression matching the
// substring of the current line that is the query's input selection.
//
// The expected output for each query is provided in the accompanying
// .golden file.
//
// (Location information is not included because it's too fragile to
// display as text.  TODO(adonovan): think about how we can test its
// correctness, since it is critical information.)
//
// Run this test with:
// 	% go test golang.org/x/tools/cmd/guru -update
// to update the golden files.

import (
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	guru "golang.org/x/tools/cmd/guru"
)

var updateFlag = flag.Bool("update", false, "Update the golden files.")

type query struct {
	id       string         // unique id
	verb     string         // query mode, e.g. "callees"
	posn     token.Position // query position
	filename string
	queryPos string // query position in command-line syntax
}

func parseRegexp(text string) (*regexp.Regexp, error) {
	pattern, err := strconv.Unquote(text)
	if err != nil {
		return nil, fmt.Errorf("can't unquote %s", text)
	}
	return regexp.Compile(pattern)
}

// parseQueries parses and returns the queries in the named file.
func parseQueries(t *testing.T, filename string) []*query {
	filedata, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	// Parse the file once to discover the test queries.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, filedata, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	lines := bytes.Split(filedata, []byte("\n"))

	var queries []*query
	queriesById := make(map[string]*query)

	// Find all annotations of these forms:
	expectRe := regexp.MustCompile(`@([a-z]+)\s+(\S+)\s+(\".*)$`) // @verb id "regexp"
	for _, c := range f.Comments {
		text := strings.TrimSpace(c.Text())
		if text == "" || text[0] != '@' {
			continue
		}
		posn := fset.Position(c.Pos())

		// @verb id "regexp"
		match := expectRe.FindStringSubmatch(text)
		if match == nil {
			t.Errorf("%s: ill-formed query: %s", posn, text)
			continue
		}

		id := match[2]
		if prev, ok := queriesById[id]; ok {
			t.Errorf("%s: duplicate id %s", posn, id)
			t.Errorf("%s: previously used here", prev.posn)
			continue
		}

		q := &query{
			id:       id,
			verb:     match[1],
			filename: filename,
			posn:     posn,
		}

		if match[3] != `"nopos"` {
			selectRe, err := parseRegexp(match[3])
			if err != nil {
				t.Errorf("%s: %s", posn, err)
				continue
			}

			// Find text of the current line, sans query.
			// (Queries must be // not /**/ comments.)
			line := lines[posn.Line-1][:posn.Column-1]

			// Apply regexp to current line to find input selection.
			loc := selectRe.FindIndex(line)
			if loc == nil {
				t.Errorf("%s: selection pattern %s doesn't match line %q",
					posn, match[3], string(line))
				continue
			}

			// Assumes ASCII. TODO(adonovan): test on UTF-8.
			linestart := posn.Offset - (posn.Column - 1)

			// Compute the file offsets.
			q.queryPos = fmt.Sprintf("%s:#%d,#%d",
				filename, linestart+loc[0], linestart+loc[1])
		}

		queries = append(queries, q)
		queriesById[id] = q
	}

	// Return the slice, not map, for deterministic iteration.
	return queries
}

// doQuery poses query q to the guru and writes its response and
// error (if any) to out.
func doQuery(out io.Writer, q *query, json bool) {
	fmt.Fprintf(out, "-------- @%s %s --------\n", q.verb, q.id)

	var buildContext = build.Default
	buildContext.GOPATH = "testdata"
	pkg := filepath.Dir(strings.TrimPrefix(q.filename, "testdata/src/"))

	gopathAbs, _ := filepath.Abs(buildContext.GOPATH)

	var outputMu sync.Mutex // guards outputs
	var outputs []string    // JSON objects or lines of text
	outputFn := func(fset *token.FileSet, qr guru.QueryResult) {
		outputMu.Lock()
		defer outputMu.Unlock()
		if json {
			jsonstr := string(qr.JSON(fset))
			// Sanitize any absolute filenames that creep in.
			jsonstr = strings.Replace(jsonstr, gopathAbs, "$GOPATH", -1)
			outputs = append(outputs, jsonstr)
		} else {
			// suppress position information
			qr.PrintPlain(func(_ interface{}, format string, args ...interface{}) {
				outputs = append(outputs, fmt.Sprintf(format, args...))
			})
		}
	}

	query := guru.Query{
		Pos:        q.queryPos,
		Build:      &buildContext,
		Scope:      []string{pkg},
		Reflection: true,
		Output:     outputFn,
	}

	if err := guru.Run(q.verb, &query); err != nil {
		fmt.Fprintf(out, "\nError: %s\n", err)
		return
	}

	// In a "referrers" query, references are sorted within each
	// package but packages are visited in arbitrary order,
	// so for determinism we sort them.  Line 0 is a caption.
	if q.verb == "referrers" {
		sort.Strings(outputs[1:])
	}

	for _, output := range outputs {
		fmt.Fprintf(out, "%s\n", output)
	}

	if !json {
		io.WriteString(out, "\n")
	}
}

func TestGuru(t *testing.T) {
	switch runtime.GOOS {
	case "android":
		t.Skipf("skipping test on %q (no testdata dir)", runtime.GOOS)
	case "windows":
		t.Skipf("skipping test on %q (no /usr/bin/diff)", runtime.GOOS)
	}

	for _, filename := range []string{
		"testdata/src/calls/main.go",
		"testdata/src/describe/main.go",
		"testdata/src/freevars/main.go",
		"testdata/src/implements/main.go",
		"testdata/src/implements-methods/main.go",
		"testdata/src/imports/main.go",
		"testdata/src/peers/main.go",
		"testdata/src/pointsto/main.go",
		"testdata/src/referrers/main.go",
		"testdata/src/reflection/main.go",
		"testdata/src/what/main.go",
		"testdata/src/whicherrs/main.go",
		// JSON:
		// TODO(adonovan): most of these are very similar; combine them.
		"testdata/src/calls-json/main.go",
		"testdata/src/peers-json/main.go",
		"testdata/src/definition-json/main.go",
		"testdata/src/describe-json/main.go",
		"testdata/src/implements-json/main.go",
		"testdata/src/implements-methods-json/main.go",
		"testdata/src/pointsto-json/main.go",
		"testdata/src/referrers-json/main.go",
		"testdata/src/what-json/main.go",
	} {
		json := strings.Contains(filename, "-json/")
		queries := parseQueries(t, filename)
		golden := filename + "lden"
		got := filename + "t"
		gotfh, err := os.Create(got)
		if err != nil {
			t.Errorf("Create(%s) failed: %s", got, err)
			continue
		}
		defer gotfh.Close()
		defer os.Remove(got)

		// Run the guru on each query, redirecting its output
		// and error (if any) to the foo.got file.
		for _, q := range queries {
			doQuery(gotfh, q, json)
		}

		// Compare foo.got with foo.golden.
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "plan9":
			cmd = exec.Command("/bin/diff", "-c", golden, got)
		default:
			cmd = exec.Command("/usr/bin/diff", "-u", golden, got)
		}
		buf := new(bytes.Buffer)
		cmd.Stdout = buf
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Errorf("Guru tests for %s failed: %s.\n%s\n",
				filename, err, buf)

			if *updateFlag {
				t.Logf("Updating %s...", golden)
				if err := exec.Command("/bin/cp", got, golden).Run(); err != nil {
					t.Errorf("Update failed: %s", err)
				}
			}
		}
	}
}

func TestIssue14684(t *testing.T) {
	var buildContext = build.Default
	buildContext.GOPATH = "testdata"
	query := guru.Query{
		Pos:   "testdata/src/README.txt:#1",
		Build: &buildContext,
	}
	err := guru.Run("freevars", &query)
	if err == nil {
		t.Fatal("guru query succeeded unexpectedly")
	}
	if got, want := err.Error(), "testdata/src/README.txt is not a Go source file"; got != want {
		t.Errorf("query error was %q, want %q", got, want)
	}
}
