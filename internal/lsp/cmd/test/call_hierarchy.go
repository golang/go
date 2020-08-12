// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
)

func (r *runner) CallHierarchy(t *testing.T, spn span.Span, expectedCalls *tests.CallHierarchyResult) {
	collectCallSpansString := func(callItems []protocol.CallHierarchyItem) string {
		var callSpans []string
		for _, call := range callItems {
			mapper, err := r.data.Mapper(call.URI.SpanURI())
			if err != nil {
				t.Fatal(err)
			}
			callSpan, err := mapper.Span(protocol.Location{URI: call.URI, Range: call.Range})
			if err != nil {
				t.Fatal(err)
			}
			callSpans = append(callSpans, fmt.Sprint(callSpan))
		}
		// to make tests deterministic
		sort.Strings(callSpans)
		return r.Normalize(strings.Join(callSpans, "\n"))
	}

	expectIn, expectOut := collectCallSpansString(expectedCalls.IncomingCalls), collectCallSpansString(expectedCalls.OutgoingCalls)
	expectIdent := r.Normalize(fmt.Sprint(spn))

	uri := spn.URI()
	filename := uri.Filename()
	target := filename + fmt.Sprintf(":%v:%v", spn.Start().Line(), spn.Start().Column())

	got, stderr := r.NormalizeGoplsCmd(t, "call_hierarchy", target)
	if stderr != "" {
		t.Fatalf("call_hierarchy failed for %s: %s", target, stderr)
	}

	gotIn, gotIdent, gotOut := cleanCallHierarchyCmdResult(got)
	if expectIn != gotIn {
		t.Errorf("incoming calls call_hierarchy failed for %s expected:\n%s\ngot:\n%s", target, expectIn, gotIn)
	}
	if expectIdent != gotIdent {
		t.Errorf("call_hierarchy failed for %s expected:\n%s\ngot:\n%s", target, expectIdent, gotIdent)
	}
	if expectOut != gotOut {
		t.Errorf("outgoing calls call_hierarchy failed for %s expected:\n%s\ngot:\n%s", target, expectOut, gotOut)
	}

}

// parses function URI and Range from call hierarchy cmd output to
// incoming, identifier and outgoing calls (returned in that order)
// ex: "identifier: function d at .../callhierarchy/callhierarchy.go:19:6-7" -> ".../callhierarchy/callhierarchy.go:19:6-7"
func cleanCallHierarchyCmdResult(output string) (incoming, ident, outgoing string) {
	var incomingCalls, outgoingCalls []string
	for _, out := range strings.Split(output, "\n") {
		if out == "" {
			continue
		}

		callLocation := out[strings.LastIndex(out, " ")+1:]
		if strings.HasPrefix(out, "caller") {
			incomingCalls = append(incomingCalls, callLocation)
		} else if strings.HasPrefix(out, "callee") {
			outgoingCalls = append(outgoingCalls, callLocation)
		} else {
			ident = callLocation
		}
	}
	sort.Strings(incomingCalls)
	sort.Strings(outgoingCalls)
	incoming, outgoing = strings.Join(incomingCalls, "\n"), strings.Join(outgoingCalls, "\n")
	return
}
