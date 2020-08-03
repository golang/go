// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
)

func (r *runner) CallHierarchy(t *testing.T, spn span.Span, expectedCalls *tests.CallHierarchyResult) {
	var result []string
	// TODO: add expectedCalls.IncomingCalls and expectedCalls.OutgoingCalls to this array once implemented
	result = append(result, fmt.Sprint(spn))

	sort.Strings(result) // to make tests deterministic
	expect := r.Normalize(strings.Join(result, "\n"))

	uri := spn.URI()
	filename := uri.Filename()
	target := filename + fmt.Sprintf(":%v:%v", spn.Start().Line(), spn.Start().Column())

	got, stderr := r.NormalizeGoplsCmd(t, "call_hierarchy", target)
	got = cleanCallHierarchyCmdResult(got)
	if stderr != "" {
		t.Errorf("call_hierarchy failed for %s: %s", target, stderr)
	} else if expect != got {
		t.Errorf("call_hierarchy failed for %s expected:\n%s\ngot:\n%s", target, expect, got)
	}
}

// removes all info except URI and Range from printed output and sorts the result
// ex: "identifier: func() d at file://callhierarchy/callhierarchy.go:19:6-7" -> "file://callhierarchy/callhierarchy.go:19:6-7"
func cleanCallHierarchyCmdResult(output string) string {
	var clean []string
	for _, out := range strings.Split(output, "\n") {
		if out == "" {
			continue
		}
		clean = append(clean, out[strings.LastIndex(out, " ")+1:])
	}
	sort.Strings(clean)
	return strings.Join(clean, "\n")
}
