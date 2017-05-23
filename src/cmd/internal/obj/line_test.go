// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"testing"
)

func TestLineHist(t *testing.T) {
	ctxt := new(Link)
	ctxt.Hash = make(map[SymVer]*LSym)

	ctxt.LineHist.Push(1, "a.c")
	ctxt.LineHist.Push(3, "a.h")
	ctxt.LineHist.Pop(5)
	ctxt.LineHist.Update(7, "linedir", 2)
	ctxt.LineHist.Pop(9)
	ctxt.LineHist.Push(11, "b.c")
	ctxt.LineHist.Pop(13)

	var expect = []string{
		0:  "??:0",
		1:  "a.c:1",
		2:  "a.c:2",
		3:  "a.h:1",
		4:  "a.h:2",
		5:  "a.c:3",
		6:  "a.c:4",
		7:  "linedir:2",
		8:  "linedir:3",
		9:  "??:0",
		10: "??:0",
		11: "b.c:1",
		12: "b.c:2",
		13: "??:0",
		14: "??:0",
	}

	for i, want := range expect {
		f, l := linkgetline(ctxt, int32(i))
		have := fmt.Sprintf("%s:%d", f.Name, l)
		if have != want {
			t.Errorf("linkgetline(%d) = %q, want %q", i, have, want)
		}
	}
}
