// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestUnify(t *testing.T) {
	paths, err := filepath.Glob("testdata/*")
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("no testdata found")
	}
	for _, path := range paths {
		// Skip paths starting with _ so experimental files can be added.
		base := filepath.Base(path)
		if base[0] == '_' {
			continue
		}
		if !strings.HasSuffix(base, ".yaml") {
			t.Errorf("non-.yaml file in testdata: %s", base)
			continue
		}
		base = strings.TrimSuffix(base, ".yaml")

		t.Run(base, func(t *testing.T) {
			testUnify(t, path)
		})
	}
}

func testUnify(t *testing.T, path string) {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	type testCase struct {
		Skip  bool
		Name  string
		Unify []Closure
		Want  yaml.Node
		All   yaml.Node
	}
	dec := yaml.NewDecoder(f)

	for i := 0; ; i++ {
		var tc testCase
		err := dec.Decode(&tc)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}

		name := tc.Name
		if name == "" {
			name = fmt.Sprint(i)
		}

		t.Run(name, func(t *testing.T) {
			if tc.Skip {
				t.Skip("skip: true set in test case")
			}

			defer func() {
				p := recover()
				if p != nil || t.Failed() {
					// Redo with a trace
					//
					// TODO: Use t.Output() in Go 1.25.
					var buf bytes.Buffer
					Debug.UnifyLog = &buf
					func() {
						defer func() {
							// If the original unify panicked, the second one
							// probably will, too. Ignore it and let the first panic
							// bubble.
							recover()
						}()
						Unify(tc.Unify...)
					}()
					Debug.UnifyLog = nil
					t.Logf("Trace:\n%s", buf.String())
				}
				if p != nil {
					panic(p)
				}
			}()

			// Unify the test cases
			//
			// TODO: Try reordering the inputs also
			c, err := Unify(tc.Unify...)
			if err != nil {
				// TODO: Tests of errors
				t.Fatal(err)
			}

			// Encode the result back to YAML so we can check if it's structurally
			// equal.
			clean := func(val any) *yaml.Node {
				var node yaml.Node
				node.Encode(val)
				for n := range allYamlNodes(&node) {
					// Canonicalize the style. There may be other style flags we need to
					// muck with.
					n.Style &^= yaml.FlowStyle
					n.HeadComment = ""
					n.LineComment = ""
					n.FootComment = ""
				}
				return &node
			}
			check := func(gotVal any, wantNode *yaml.Node) {
				got, err := yaml.Marshal(clean(gotVal))
				if err != nil {
					t.Fatalf("Encoding Value back to yaml failed: %s", err)
				}
				want, err := yaml.Marshal(clean(wantNode))
				if err != nil {
					t.Fatalf("Encoding Want back to yaml failed: %s", err)
				}

				if !bytes.Equal(got, want) {
					t.Errorf("%s:%d:\nwant:\n%sgot\n%s", f.Name(), wantNode.Line, want, got)
				}
			}
			if tc.Want.Kind != 0 {
				check(c.val, &tc.Want)
			}
			if tc.All.Kind != 0 {
				fVal := slices.Collect(c.All())
				check(fVal, &tc.All)
			}
		})
	}
}
