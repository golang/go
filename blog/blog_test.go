// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blog

import (
	"bytes"
	"testing"
)

func TestLinkRewrite(t *testing.T) {
	tests := []struct {
		input  string
		output string
	}{
		{
			`For instance, the <a href="https://golang.org/pkg/bytes/" target="_blank">bytes package</a> from the standard library exports the <code>Buffer</code> type.`,
			`For instance, the <a href="/pkg/bytes/" target="_blank">bytes package</a> from the standard library exports the <code>Buffer</code> type.`},
		{
			`(The <a href="https://golang.org/cmd/gofmt/" target="_blank">gofmt command</a> has a <code>-r</code> flag that provides a syntax-aware search and replace, making large-scale refactoring easier.)`,
			`(The <a href="/cmd/gofmt/" target="_blank">gofmt command</a> has a <code>-r</code> flag that provides a syntax-aware search and replace, making large-scale refactoring easier.)`,
		},
		{
			`<a href="//golang.org/LICENSE">BSD license</a>.<br> <a href="//golang.org/doc/tos.html">Terms of Service</a> `,
			`<a href="//golang.org/LICENSE">BSD license</a>.<br> <a href="//golang.org/doc/tos.html">Terms of Service</a> `,
		},
		{
			`For instance, the <code>websocket</code> package from the <code>go.net</code> sub-repository has an import path of <code>&#34;golang.org/x/net/websocket&#34;</code>.`,
			`For instance, the <code>websocket</code> package from the <code>go.net</code> sub-repository has an import path of <code>&#34;golang.org/x/net/websocket&#34;</code>.`,
		},
	}
	for _, test := range tests {
		var buf bytes.Buffer
		_, err := golangOrgAbsLinkReplacer.WriteString(&buf, test.input)
		if err != nil {
			t.Errorf("unexpected error during replacing links. Got: %#v, Want: nil.\n", err)
			continue
		}
		if got, want := buf.String(), test.output; got != want {
			t.Errorf("WriteString(%q) = %q. Expected: %q", test.input, got, want)
		}
	}
}
