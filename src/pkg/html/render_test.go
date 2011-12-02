// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"testing"
)

func TestRenderer(t *testing.T) {
	n := &Node{
		Type: ElementNode,
		Data: "html",
		Child: []*Node{
			{
				Type: ElementNode,
				Data: "head",
			},
			{
				Type: ElementNode,
				Data: "body",
				Child: []*Node{
					{
						Type: TextNode,
						Data: "0<1",
					},
					{
						Type: ElementNode,
						Data: "p",
						Attr: []Attribute{
							{
								Key: "id",
								Val: "A",
							},
							{
								Key: "foo",
								Val: `abc"def`,
							},
						},
						Child: []*Node{
							{
								Type: TextNode,
								Data: "2",
							},
							{
								Type: ElementNode,
								Data: "b",
								Attr: []Attribute{
									{
										Key: "empty",
										Val: "",
									},
								},
								Child: []*Node{
									{
										Type: TextNode,
										Data: "3",
									},
								},
							},
							{
								Type: ElementNode,
								Data: "i",
								Attr: []Attribute{
									{
										Key: "backslash",
										Val: `\`,
									},
								},
								Child: []*Node{
									{
										Type: TextNode,
										Data: "&4",
									},
								},
							},
						},
					},
					{
						Type: TextNode,
						Data: "5",
					},
					{
						Type: ElementNode,
						Data: "blockquote",
					},
					{
						Type: ElementNode,
						Data: "br",
					},
					{
						Type: TextNode,
						Data: "6",
					},
				},
			},
		},
	}
	want := `<html><head></head><body>0&lt;1<p id="A" foo="abc&quot;def">` +
		`2<b empty="">3</b><i backslash="\">&amp;4</i></p>` +
		`5<blockquote></blockquote><br/>6</body></html>`
	b := new(bytes.Buffer)
	if err := Render(b, n); err != nil {
		t.Fatal(err)
	}
	if got := b.String(); got != want {
		t.Errorf("got vs want:\n%s\n%s\n", got, want)
	}
}
