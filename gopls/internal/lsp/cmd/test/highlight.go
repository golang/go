// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"testing"

	"fmt"

	"golang.org/x/tools/gopls/internal/span"
)

func (r *runner) Highlight(t *testing.T, spn span.Span, spans []span.Span) {
	var expect string
	for _, l := range spans {
		expect += fmt.Sprintln(l)
	}
	expect = r.Normalize(expect)

	uri := spn.URI()
	filename := uri.Filename()
	target := filename + ":" + fmt.Sprint(spn.Start().Line()) + ":" + fmt.Sprint(spn.Start().Column())
	got, _ := r.NormalizeGoplsCmd(t, "highlight", target)
	if expect != got {
		t.Errorf("highlight failed for %s expected:\n%s\ngot:\n%s", target, expect, got)
	}
}
