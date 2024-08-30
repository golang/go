// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"cmd/internal/src"
	"fmt"
	"testing"
)

func TestGetFileSymbolAndLine(t *testing.T) {
	ctxt := new(Link)
	ctxt.hash = make(map[string]*LSym)
	ctxt.statichash = make(map[string]*LSym)

	afile := src.NewFileBase("a.go", "a.go")
	bfile := src.NewFileBase("b.go", "/foo/bar/b.go")
	lfile := src.NewLinePragmaBase(src.MakePos(afile, 8, 1), "linedir", "linedir", 100, 1)

	var tests = []struct {
		pos  src.Pos
		want string
	}{
		{src.NoPos, "??:0"},
		{src.MakePos(afile, 1, 0), "a.go:1"},
		{src.MakePos(afile, 2, 0), "a.go:2"},
		{src.MakePos(bfile, 10, 4), "/foo/bar/b.go:10"},
		{src.MakePos(lfile, 10, 0), "linedir:102"}, // 102 == 100 + (10 - (7+1))
	}

	for _, test := range tests {
		fileIndex, line := ctxt.getFileIndexAndLine(ctxt.PosTable.XPos(test.pos))

		file := "??"
		if fileIndex >= 0 {
			file = ctxt.PosTable.FileTable()[fileIndex]
		}
		got := fmt.Sprintf("%s:%d", file, line)

		if got != test.want {
			t.Errorf("ctxt.getFileSymbolAndLine(%v) = %q, want %q", test.pos, got, test.want)
		}
	}
}
