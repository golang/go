// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func TestApplyEdits(t *testing.T) {
	tests := []struct {
		label   string
		content string
		edits   []protocol.TextEdit
		want    string
		wantErr bool
	}{
		{
			label: "empty content",
		},
		{
			label:   "empty edit",
			content: "hello",
			edits:   []protocol.TextEdit{},
			want:    "hello",
		},
		{
			label:   "unicode edit",
			content: "hello, 日本語",
			edits: []protocol.TextEdit{
				NewEdit(0, 7, 0, 10, "world"),
			},
			want: "hello, world",
		},
		{
			label:   "range edit",
			content: "ABC\nDEF\nGHI\nJKL",
			edits: []protocol.TextEdit{
				NewEdit(1, 1, 2, 3, "12\n345"),
			},
			want: "ABC\nD12\n345\nJKL",
		},
		{
			label:   "regression test for issue #57627",
			content: "go 1.18\nuse moda/a",
			edits: []protocol.TextEdit{
				NewEdit(1, 0, 1, 0, "\n"),
				NewEdit(2, 0, 2, 0, "\n"),
			},
			want: "go 1.18\n\nuse moda/a\n",
		},
		{
			label:   "end before start",
			content: "ABC\nDEF\nGHI\nJKL",
			edits: []protocol.TextEdit{
				NewEdit(2, 3, 1, 1, "12\n345"),
			},
			wantErr: true,
		},
		{
			label:   "out of bounds line",
			content: "ABC\nDEF\nGHI\nJKL",
			edits: []protocol.TextEdit{
				NewEdit(1, 1, 4, 3, "12\n345"),
			},
			wantErr: true,
		},
		{
			label:   "out of bounds column",
			content: "ABC\nDEF\nGHI\nJKL",
			edits: []protocol.TextEdit{
				NewEdit(1, 4, 2, 3, "12\n345"),
			},
			wantErr: true,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.label, func(t *testing.T) {
			got, err := applyEdits(protocol.NewMapper("", []byte(test.content)), test.edits, false)
			if (err != nil) != test.wantErr {
				t.Errorf("got err %v, want error: %t", err, test.wantErr)
			}
			if err != nil {
				return
			}
			if got := string(got); got != test.want {
				t.Errorf("got %q, want %q", got, test.want)
			}
		})
	}
}
