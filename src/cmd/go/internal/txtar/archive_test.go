// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txtar

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

var tests = []struct {
	name   string
	text   string
	parsed *Archive
}{
	{
		name: "basic",
		text: `comment1
comment2
-- file1 --
File 1 text.
-- foo ---
More file 1 text.
-- file 2 --
File 2 text.
-- empty --
-- noNL --
hello world`,
		parsed: &Archive{
			Comment: []byte("comment1\ncomment2\n"),
			Files: []File{
				{"file1", []byte("File 1 text.\n-- foo ---\nMore file 1 text.\n")},
				{"file 2", []byte("File 2 text.\n")},
				{"empty", []byte{}},
				{"noNL", []byte("hello world\n")},
			},
		},
	},
}

func Test(t *testing.T) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := Parse([]byte(tt.text))
			if !reflect.DeepEqual(a, tt.parsed) {
				t.Fatalf("Parse: wrong output:\nhave:\n%s\nwant:\n%s", shortArchive(a), shortArchive(tt.parsed))
			}
			text := Format(a)
			a = Parse(text)
			if !reflect.DeepEqual(a, tt.parsed) {
				t.Fatalf("Parse after Format: wrong output:\nhave:\n%s\nwant:\n%s", shortArchive(a), shortArchive(tt.parsed))
			}
		})
	}
}

func shortArchive(a *Archive) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "comment: %q\n", a.Comment)
	for _, f := range a.Files {
		fmt.Fprintf(&buf, "file %q: %q\n", f.Name, f.Data)
	}
	return buf.String()
}
