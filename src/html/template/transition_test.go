// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"strings"
	"testing"
)

func TestFindEndTag(t *testing.T) {
	tests := []struct {
		s, tag string
		want   int
	}{
		{"", "tag", -1},
		{"hello </textarea> hello", "textarea", 6},
		{"hello </TEXTarea> hello", "textarea", 6},
		{"hello </textAREA>", "textarea", 6},
		{"hello </textarea", "textareax", -1},
		{"hello </textarea>", "tag", -1},
		{"hello tag </textarea", "tag", -1},
		{"hello </tag> </other> </textarea> <other>", "textarea", 22},
		{"</textarea> <other>", "textarea", 0},
		{"<div> </div> </TEXTAREA>", "textarea", 13},
		{"<div> </div> </TEXTAREA\t>", "textarea", 13},
		{"<div> </div> </TEXTAREA >", "textarea", 13},
		{"<div> </div> </TEXTAREAfoo", "textarea", -1},
		{"</TEXTAREAfoo </textarea>", "textarea", 14},
		{"<</script >", "script", 1},
		{"</script>", "textarea", -1},
	}
	for _, test := range tests {
		if got := indexTagEnd([]byte(test.s), []byte(test.tag)); test.want != got {
			t.Errorf("%q/%q: want\n\t%d\nbut got\n\t%d", test.s, test.tag, test.want, got)
		}
	}
}

func BenchmarkTemplateSpecialTags(b *testing.B) {

	r := struct {
		Name, Gift string
	}{"Aunt Mildred", "bone china tea set"}

	h1 := "<textarea> Hello Hello Hello </textarea> "
	h2 := "<textarea> <p> Dear {{.Name}},\n{{with .Gift}}Thank you for the lovely {{.}}. {{end}}\nBest wishes. </p>\n</textarea>"
	html := strings.Repeat(h1, 100) + h2 + strings.Repeat(h1, 100) + h2

	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		tmpl := Must(New("foo").Parse(html))
		if err := tmpl.Execute(&buf, r); err != nil {
			b.Fatal(err)
		}
		buf.Reset()
	}
}
