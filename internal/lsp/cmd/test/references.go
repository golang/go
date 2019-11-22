// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"sort"
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) References(t *testing.T, spn span.Span, itemList []span.Span) {
	var itemStrings []string
	for _, i := range itemList {
		itemStrings = append(itemStrings, fmt.Sprint(i))
	}
	sort.Strings(itemStrings)
	var expect string
	for _, i := range itemStrings {
		expect += i + "\n"
	}
	expect = r.Normalize(expect)

	uri := spn.URI()
	filename := uri.Filename()
	target := filename + fmt.Sprintf(":%v:%v", spn.Start().Line(), spn.Start().Column())
	got, _ := r.NormalizeGoplsCmd(t, "references", target)
	if expect != got {
		t.Errorf("references failed for %s expected:\n%s\ngot:\n%s", target, expect, got)
	}
}
