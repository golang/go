// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expect_test

import (
	"bytes"
	"go/token"
	"io/ioutil"
	"testing"

	"golang.org/x/tools/go/expect"
)

func TestMarker(t *testing.T) {
	for _, tt := range []struct {
		filename      string
		expectNotes   int
		expectMarkers map[string]string
		expectChecks  map[string][]interface{}
	}{
		{
			filename:    "testdata/test.go",
			expectNotes: 13,
			expectMarkers: map[string]string{
				"αSimpleMarker": "α",
				"OffsetMarker":  "β",
				"RegexMarker":   "γ",
				"εMultiple":     "ε",
				"ζMarkers":      "ζ",
				"ηBlockMarker":  "η",
				"Declared":      "η",
				"Comment":       "ι",
				"LineComment":   "someFunc",
				"NonIdentifier": "+",
				"StringMarker":  "\"hello\"",
			},
			expectChecks: map[string][]interface{}{
				"αSimpleMarker": nil,
				"StringAndInt":  {"Number %d", int64(12)},
				"Bool":          {true},
			},
		},
		{
			filename:    "testdata/go.fake.mod",
			expectNotes: 2,
			expectMarkers: map[string]string{
				"αMarker": "αfake1α",
				"βMarker": "require golang.org/modfile v0.0.0",
			},
		},
	} {
		t.Run(tt.filename, func(t *testing.T) {
			content, err := ioutil.ReadFile(tt.filename)
			if err != nil {
				t.Fatal(err)
			}
			readFile := func(string) ([]byte, error) { return content, nil }

			markers := make(map[string]token.Pos)
			for name, tok := range tt.expectMarkers {
				offset := bytes.Index(content, []byte(tok))
				markers[name] = token.Pos(offset + 1)
				end := bytes.Index(content[offset:], []byte(tok))
				if end > 0 {
					markers[name+"@"] = token.Pos(offset + end + 2)
				}
			}

			fset := token.NewFileSet()
			notes, err := expect.Parse(fset, tt.filename, content)
			if err != nil {
				t.Fatalf("Failed to extract notes: %v", err)
			}
			if len(notes) != tt.expectNotes {
				t.Errorf("Expected %v notes, got %v", tt.expectNotes, len(notes))
			}
			for _, n := range notes {
				switch {
				case n.Args == nil:
					// A //@foo note associates the name foo with the position of the
					// first match of "foo" on the current line.
					checkMarker(t, fset, readFile, markers, n.Pos, n.Name, n.Name)
				case n.Name == "mark":
					// A //@mark(name, "pattern") note associates the specified name
					// with the position on the first match of pattern on the current line.
					if len(n.Args) != 2 {
						t.Errorf("%v: expected 2 args to mark, got %v", fset.Position(n.Pos), len(n.Args))
						continue
					}
					ident, ok := n.Args[0].(expect.Identifier)
					if !ok {
						t.Errorf("%v: identifier, got %T", fset.Position(n.Pos), n.Args[0])
						continue
					}
					checkMarker(t, fset, readFile, markers, n.Pos, string(ident), n.Args[1])

				case n.Name == "check":
					// A //@check(args, ...) note specifies some hypothetical action to
					// be taken by the test driver and its expected outcome.
					// In this test, the action is to compare the arguments
					// against expectChecks.
					if len(n.Args) < 1 {
						t.Errorf("%v: expected 1 args to check, got %v", fset.Position(n.Pos), len(n.Args))
						continue
					}
					ident, ok := n.Args[0].(expect.Identifier)
					if !ok {
						t.Errorf("%v: identifier, got %T", fset.Position(n.Pos), n.Args[0])
						continue
					}
					args, ok := tt.expectChecks[string(ident)]
					if !ok {
						t.Errorf("%v: unexpected check %v", fset.Position(n.Pos), ident)
						continue
					}
					if len(n.Args) != len(args)+1 {
						t.Errorf("%v: expected %v args to check, got %v", fset.Position(n.Pos), len(args)+1, len(n.Args))
						continue
					}
					for i, got := range n.Args[1:] {
						if args[i] != got {
							t.Errorf("%v: arg %d expected %v, got %v", fset.Position(n.Pos), i, args[i], got)
						}
					}
				default:
					t.Errorf("Unexpected note %v at %v", n.Name, fset.Position(n.Pos))
				}
			}
		})
	}
}

func checkMarker(t *testing.T, fset *token.FileSet, readFile expect.ReadFile, markers map[string]token.Pos, pos token.Pos, name string, pattern interface{}) {
	start, end, err := expect.MatchBefore(fset, readFile, pos, pattern)
	if err != nil {
		t.Errorf("%v: MatchBefore failed: %v", fset.Position(pos), err)
		return
	}
	if start == token.NoPos {
		t.Errorf("%v: Pattern %v did not match", fset.Position(pos), pattern)
		return
	}
	expectStart, ok := markers[name]
	if !ok {
		t.Errorf("%v: unexpected marker %v", fset.Position(pos), name)
		return
	}
	if start != expectStart {
		t.Errorf("%v: Expected %v got %v", fset.Position(pos), fset.Position(expectStart), fset.Position(start))
	}
	if expectEnd, ok := markers[name+"@"]; ok && end != expectEnd {
		t.Errorf("%v: Expected end %v got %v", fset.Position(pos), fset.Position(expectEnd), fset.Position(end))
	}
}
