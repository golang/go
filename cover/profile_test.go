// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cover

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
)

func TestParseProfiles(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		output    []*Profile
		expectErr bool
	}{
		{
			name:   "parsing an empty file produces empty output",
			input:  `mode: set`,
			output: []*Profile{},
		},
		{
			name: "simple valid file produces expected output",
			input: `mode: set
some/fancy/path:42.69,44.16 2 1`,
			output: []*Profile{
				{
					FileName: "some/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 1,
						},
					},
				},
			},
		},
		{
			name: "file with syntax characters in path produces expected output",
			input: `mode: set
some fancy:path/some,file.go:42.69,44.16 2 1`,
			output: []*Profile{
				{
					FileName: "some fancy:path/some,file.go",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 1,
						},
					},
				},
			},
		},
		{
			name: "file with multiple blocks in one file produces expected output",
			input: `mode: set
some/fancy/path:42.69,44.16 2 1
some/fancy/path:44.16,46.3 1 0`,
			output: []*Profile{
				{
					FileName: "some/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 1,
						},
						{
							StartLine: 44, StartCol: 16,
							EndLine: 46, EndCol: 3,
							NumStmt: 1, Count: 0,
						},
					},
				},
			},
		},
		{
			name: "file with multiple files produces expected output",
			input: `mode: set
another/fancy/path:44.16,46.3 1 0
some/fancy/path:42.69,44.16 2 1`,
			output: []*Profile{
				{
					FileName: "another/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 44, StartCol: 16,
							EndLine: 46, EndCol: 3,
							NumStmt: 1, Count: 0,
						},
					},
				},
				{
					FileName: "some/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 1,
						},
					},
				},
			},
		},
		{
			name: "intertwined files are merged correctly",
			input: `mode: set
some/fancy/path:42.69,44.16 2 1
another/fancy/path:47.2,47.13 1 1
some/fancy/path:44.16,46.3 1 0`,
			output: []*Profile{
				{
					FileName: "another/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 47, StartCol: 2,
							EndLine: 47, EndCol: 13,
							NumStmt: 1, Count: 1,
						},
					},
				},
				{
					FileName: "some/fancy/path",
					Mode:     "set",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 1,
						},
						{
							StartLine: 44, StartCol: 16,
							EndLine: 46, EndCol: 3,
							NumStmt: 1, Count: 0,
						},
					},
				},
			},
		},
		{
			name: "duplicate blocks are merged correctly",
			input: `mode: count
some/fancy/path:42.69,44.16 2 4
some/fancy/path:42.69,44.16 2 3`,
			output: []*Profile{
				{
					FileName: "some/fancy/path",
					Mode:     "count",
					Blocks: []ProfileBlock{
						{
							StartLine: 42, StartCol: 69,
							EndLine: 44, EndCol: 16,
							NumStmt: 2, Count: 7,
						},
					},
				},
			},
		},
		{
			name:      "an invalid mode line is an error",
			input:     `mode:count`,
			expectErr: true,
		},
		{
			name: "a missing field is an error",
			input: `mode: count
some/fancy/path:42.69,44.16 2`,
			expectErr: true,
		},
		{
			name: "a missing path field is an error",
			input: `mode: count
42.69,44.16 2 3`,
			expectErr: true,
		},
		{
			name: "a non-numeric count is an error",
			input: `mode: count
42.69,44.16 2 nope`,
			expectErr: true,
		},
		{
			name: "an empty path is an error",
			input: `mode: count
:42.69,44.16 2 3`,
			expectErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, err := ioutil.TempFile("", "")
			if err != nil {
				t.Fatalf("Failed to create a temp file: %v.", err)
			}
			defer func() {
				f.Close()
				os.Remove(f.Name())
			}()
			n, err := f.WriteString(tc.input)
			if err != nil {
				t.Fatalf("Failed to write to temp file: %v", err)
			}
			if n < len(tc.input) {
				t.Fatalf("Didn't write enough bytes to temp file (wrote %d, expected %d).", n, len(tc.input))
			}
			if err := f.Sync(); err != nil {
				t.Fatalf("Failed to sync temp file: %v", err)
			}

			result, err := ParseProfiles(f.Name())
			if err != nil {
				if !tc.expectErr {
					t.Errorf("Unexpected error: %v", err)
				}
				return
			}
			if tc.expectErr {
				t.Errorf("Expected an error, but got value %q", stringifyProfileArray(result))
			}
			if !reflect.DeepEqual(result, tc.output) {
				t.Errorf("Mismatched results.\nExpected: %s\nActual:   %s", stringifyProfileArray(tc.output), stringifyProfileArray(result))
			}
		})
	}
}

func stringifyProfileArray(profiles []*Profile) string {
	deref := make([]Profile, 0, len(profiles))
	for _, p := range profiles {
		deref = append(deref, *p)
	}
	return fmt.Sprintf("%#v", deref)
}

func BenchmarkParseLine(b *testing.B) {
	const line = "k8s.io/kubernetes/cmd/kube-controller-manager/app/options/ttlafterfinishedcontroller.go:31.73,32.14 1 1"
	b.SetBytes(int64(len(line)))
	for n := 0; n < b.N; n++ {
		parseLine(line)
	}
}
