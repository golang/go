// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"testing"
	"text/template"
	"text/template/parse"
)

func TestClone(t *testing.T) {
	tests := []struct {
		input, want, wantClone string
	}{
		{
			`Hello, {{if true}}{{"<World>"}}{{end}}!`,
			"Hello, <World>!",
			"Hello, &lt;World&gt;!",
		},
		{
			`Hello, {{if false}}{{.X}}{{else}}{{"<World>"}}{{end}}!`,
			"Hello, <World>!",
			"Hello, &lt;World&gt;!",
		},
		{
			`Hello, {{with "<World>"}}{{.}}{{end}}!`,
			"Hello, <World>!",
			"Hello, &lt;World&gt;!",
		},
		{
			`{{range .}}<p>{{.}}</p>{{end}}`,
			"<p>foo</p><p><bar></p><p>baz</p>",
			"<p>foo</p><p>&lt;bar&gt;</p><p>baz</p>",
		},
		{
			`Hello, {{"<World>" | html}}!`,
			"Hello, &lt;World&gt;!",
			"Hello, &lt;World&gt;!",
		},
		{
			`Hello{{if 1}}, World{{else}}{{template "d"}}{{end}}!`,
			"Hello, World!",
			"Hello, World!",
		},
	}

	for _, test := range tests {
		s := template.Must(template.New("s").Parse(test.input))
		d := template.New("d")
		d.Tree = &parse.Tree{Name: d.Name(), Root: cloneList(s.Root)}

		if want, got := s.Root.String(), d.Root.String(); want != got {
			t.Errorf("want %q, got %q", want, got)
		}

		err := escape(d)
		if err != nil {
			t.Errorf("%q: failed to escape: %s", test.input, err)
			continue
		}

		if want, got := "s", s.Name(); want != got {
			t.Errorf("want %q, got %q", want, got)
			continue
		}
		if want, got := "d", d.Name(); want != got {
			t.Errorf("want %q, got %q", want, got)
			continue
		}

		data := []string{"foo", "<bar>", "baz"}

		// Make sure escaping d did not affect s.
		var b bytes.Buffer
		s.Execute(&b, data)
		if got := b.String(); got != test.want {
			t.Errorf("%q: want %q, got %q", test.input, test.want, got)
			continue
		}

		b.Reset()
		d.Execute(&b, data)
		if got := b.String(); got != test.wantClone {
			t.Errorf("%q: want %q, got %q", test.input, test.wantClone, got)
		}
	}
}
