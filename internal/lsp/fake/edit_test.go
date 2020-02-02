// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"strings"
	"testing"
)

func TestApplyEdit(t *testing.T) {
	tests := []struct {
		label   string
		content string
		edit    Edit
		want    string
		wantErr bool
	}{
		{
			label: "empty content",
		},
		{
			label:   "empty edit",
			content: "hello",
			edit:    Edit{},
			want:    "hello",
		},
		{
			label:   "unicode edit",
			content: "hello, 日本語",
			edit: Edit{
				Start: Pos{Line: 0, Column: 7},
				End:   Pos{Line: 0, Column: 10},
				Text:  "world",
			},
			want: "hello, world",
		},
		{
			label:   "range edit",
			content: "ABC\nDEF\nGHI\nJKL",
			edit: Edit{
				Start: Pos{Line: 1, Column: 1},
				End:   Pos{Line: 2, Column: 3},
				Text:  "12\n345",
			},
			want: "ABC\nD12\n345\nJKL",
		},
		{
			label:   "end before start",
			content: "ABC\nDEF\nGHI\nJKL",
			edit: Edit{
				End:   Pos{Line: 1, Column: 1},
				Start: Pos{Line: 2, Column: 3},
				Text:  "12\n345",
			},
			wantErr: true,
		},
		{
			label:   "out of bounds line",
			content: "ABC\nDEF\nGHI\nJKL",
			edit: Edit{
				Start: Pos{Line: 1, Column: 1},
				End:   Pos{Line: 4, Column: 3},
				Text:  "12\n345",
			},
			wantErr: true,
		},
		{
			label:   "out of bounds column",
			content: "ABC\nDEF\nGHI\nJKL",
			edit: Edit{
				Start: Pos{Line: 1, Column: 4},
				End:   Pos{Line: 2, Column: 3},
				Text:  "12\n345",
			},
			wantErr: true,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.label, func(t *testing.T) {
			lines := strings.Split(test.content, "\n")
			newLines, err := editContent(lines, test.edit)
			if (err != nil) != test.wantErr {
				t.Errorf("got err %v, want error: %t", err, test.wantErr)
			}
			if err != nil {
				return
			}
			if got := strings.Join(newLines, "\n"); got != test.want {
				t.Errorf("got %q, want %q", got, test.want)
			}
		})
	}
}
