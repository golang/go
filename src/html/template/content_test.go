// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestTypedContent(t *testing.T) {
	data := []any{
		`<b> "foo%" O'Reilly &bar;`,
		CSS(`a[href =~ "//example.com"]#foo`),
		HTML(`Hello, <b>World</b> &amp;tc!`),
		HTMLAttr(` dir="ltr"`),
		JS(`c && alert("Hello, World!");`),
		JSStr(`Hello, World & O'Reilly\u0021`),
		URL(`greeting=H%69,&addressee=(World)`),
		Srcset(`greeting=H%69,&addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`),
		URL(`,foo/,`),
	}

	// For each content sensitive escaper, see how it does on
	// each of the typed strings above.
	tests := []struct {
		// A template containing a single {{.}}.
		input string
		want  []string
	}{
		{
			`<style>{{.}} { color: blue }</style>`,
			[]string{
				`ZgotmplZ`,
				// Allowed but not escaped.
				`a[href =~ "//example.com"]#foo`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
			},
		},
		{
			`<div style="{{.}}">`,
			[]string{
				`ZgotmplZ`,
				// Allowed and HTML escaped.
				`a[href =~ &#34;//example.com&#34;]#foo`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
			},
		},
		{
			`{{.}}`,
			[]string{
				`&lt;b&gt; &#34;foo%&#34; O&#39;Reilly &amp;bar;`,
				`a[href =~ &#34;//example.com&#34;]#foo`,
				// Not escaped.
				`Hello, <b>World</b> &amp;tc!`,
				` dir=&#34;ltr&#34;`,
				`c &amp;&amp; alert(&#34;Hello, World!&#34;);`,
				`Hello, World &amp; O&#39;Reilly\u0021`,
				`greeting=H%69,&amp;addressee=(World)`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`,foo/,`,
			},
		},
		{
			`<a{{.}}>`,
			[]string{
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				// Allowed and HTML escaped.
				` dir="ltr"`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
				`ZgotmplZ`,
			},
		},
		{
			`<a title={{.}}>`,
			[]string{
				`&lt;b&gt;&#32;&#34;foo%&#34;&#32;O&#39;Reilly&#32;&amp;bar;`,
				`a[href&#32;&#61;~&#32;&#34;//example.com&#34;]#foo`,
				// Tags stripped, spaces escaped, entity not re-escaped.
				`Hello,&#32;World&#32;&amp;tc!`,
				`&#32;dir&#61;&#34;ltr&#34;`,
				`c&#32;&amp;&amp;&#32;alert(&#34;Hello,&#32;World!&#34;);`,
				`Hello,&#32;World&#32;&amp;&#32;O&#39;Reilly\u0021`,
				`greeting&#61;H%69,&amp;addressee&#61;(World)`,
				`greeting&#61;H%69,&amp;addressee&#61;(World)&#32;2x,&#32;https://golang.org/favicon.ico&#32;500.5w`,
				`,foo/,`,
			},
		},
		{
			`<a title='{{.}}'>`,
			[]string{
				`&lt;b&gt; &#34;foo%&#34; O&#39;Reilly &amp;bar;`,
				`a[href =~ &#34;//example.com&#34;]#foo`,
				// Tags stripped, entity not re-escaped.
				`Hello, World &amp;tc!`,
				` dir=&#34;ltr&#34;`,
				`c &amp;&amp; alert(&#34;Hello, World!&#34;);`,
				`Hello, World &amp; O&#39;Reilly\u0021`,
				`greeting=H%69,&amp;addressee=(World)`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`,foo/,`,
			},
		},
		{
			`<textarea>{{.}}</textarea>`,
			[]string{
				`&lt;b&gt; &#34;foo%&#34; O&#39;Reilly &amp;bar;`,
				`a[href =~ &#34;//example.com&#34;]#foo`,
				// Angle brackets escaped to prevent injection of close tags, entity not re-escaped.
				`Hello, &lt;b&gt;World&lt;/b&gt; &amp;tc!`,
				` dir=&#34;ltr&#34;`,
				`c &amp;&amp; alert(&#34;Hello, World!&#34;);`,
				`Hello, World &amp; O&#39;Reilly\u0021`,
				`greeting=H%69,&amp;addressee=(World)`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`,foo/,`,
			},
		},
		{
			`<script>alert({{.}})</script>`,
			[]string{
				`"\u003cb\u003e \"foo%\" O'Reilly \u0026bar;"`,
				`"a[href =~ \"//example.com\"]#foo"`,
				`"Hello, \u003cb\u003eWorld\u003c/b\u003e \u0026amp;tc!"`,
				`" dir=\"ltr\""`,
				// Not escaped.
				`c && alert("Hello, World!");`,
				// Escape sequence not over-escaped.
				`"Hello, World & O'Reilly\u0021"`,
				`"greeting=H%69,\u0026addressee=(World)"`,
				`"greeting=H%69,\u0026addressee=(World) 2x, https://golang.org/favicon.ico 500.5w"`,
				`",foo/,"`,
			},
		},
		{
			`<button onclick="alert({{.}})">`,
			[]string{
				`&#34;\u003cb\u003e \&#34;foo%\&#34; O&#39;Reilly \u0026bar;&#34;`,
				`&#34;a[href =~ \&#34;//example.com\&#34;]#foo&#34;`,
				`&#34;Hello, \u003cb\u003eWorld\u003c/b\u003e \u0026amp;tc!&#34;`,
				`&#34; dir=\&#34;ltr\&#34;&#34;`,
				// Not JS escaped but HTML escaped.
				`c &amp;&amp; alert(&#34;Hello, World!&#34;);`,
				// Escape sequence not over-escaped.
				`&#34;Hello, World &amp; O&#39;Reilly\u0021&#34;`,
				`&#34;greeting=H%69,\u0026addressee=(World)&#34;`,
				`&#34;greeting=H%69,\u0026addressee=(World) 2x, https://golang.org/favicon.ico 500.5w&#34;`,
				`&#34;,foo/,&#34;`,
			},
		},
		{
			`<script>alert("{{.}}")</script>`,
			[]string{
				`\u003cb\u003e \u0022foo%\u0022 O\u0027Reilly \u0026bar;`,
				`a[href =~ \u0022\/\/example.com\u0022]#foo`,
				`Hello, \u003cb\u003eWorld\u003c\/b\u003e \u0026amp;tc!`,
				` dir=\u0022ltr\u0022`,
				`c \u0026\u0026 alert(\u0022Hello, World!\u0022);`,
				// Escape sequence not over-escaped.
				`Hello, World \u0026 O\u0027Reilly\u0021`,
				`greeting=H%69,\u0026addressee=(World)`,
				`greeting=H%69,\u0026addressee=(World) 2x, https:\/\/golang.org\/favicon.ico 500.5w`,
				`,foo\/,`,
			},
		},
		{
			`<script type="text/javascript">alert("{{.}}")</script>`,
			[]string{
				`\u003cb\u003e \u0022foo%\u0022 O\u0027Reilly \u0026bar;`,
				`a[href =~ \u0022\/\/example.com\u0022]#foo`,
				`Hello, \u003cb\u003eWorld\u003c\/b\u003e \u0026amp;tc!`,
				` dir=\u0022ltr\u0022`,
				`c \u0026\u0026 alert(\u0022Hello, World!\u0022);`,
				// Escape sequence not over-escaped.
				`Hello, World \u0026 O\u0027Reilly\u0021`,
				`greeting=H%69,\u0026addressee=(World)`,
				`greeting=H%69,\u0026addressee=(World) 2x, https:\/\/golang.org\/favicon.ico 500.5w`,
				`,foo\/,`,
			},
		},
		{
			`<script type="text/javascript">alert({{.}})</script>`,
			[]string{
				`"\u003cb\u003e \"foo%\" O'Reilly \u0026bar;"`,
				`"a[href =~ \"//example.com\"]#foo"`,
				`"Hello, \u003cb\u003eWorld\u003c/b\u003e \u0026amp;tc!"`,
				`" dir=\"ltr\""`,
				// Not escaped.
				`c && alert("Hello, World!");`,
				// Escape sequence not over-escaped.
				`"Hello, World & O'Reilly\u0021"`,
				`"greeting=H%69,\u0026addressee=(World)"`,
				`"greeting=H%69,\u0026addressee=(World) 2x, https://golang.org/favicon.ico 500.5w"`,
				`",foo/,"`,
			},
		},
		{
			// Not treated as JS. The output is same as for <div>{{.}}</div>
			`<script type="text/template">{{.}}</script>`,
			[]string{
				`&lt;b&gt; &#34;foo%&#34; O&#39;Reilly &amp;bar;`,
				`a[href =~ &#34;//example.com&#34;]#foo`,
				// Not escaped.
				`Hello, <b>World</b> &amp;tc!`,
				` dir=&#34;ltr&#34;`,
				`c &amp;&amp; alert(&#34;Hello, World!&#34;);`,
				`Hello, World &amp; O&#39;Reilly\u0021`,
				`greeting=H%69,&amp;addressee=(World)`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`,foo/,`,
			},
		},
		{
			`<button onclick='alert("{{.}}")'>`,
			[]string{
				`\u003cb\u003e \u0022foo%\u0022 O\u0027Reilly \u0026bar;`,
				`a[href =~ \u0022\/\/example.com\u0022]#foo`,
				`Hello, \u003cb\u003eWorld\u003c\/b\u003e \u0026amp;tc!`,
				` dir=\u0022ltr\u0022`,
				`c \u0026\u0026 alert(\u0022Hello, World!\u0022);`,
				// Escape sequence not over-escaped.
				`Hello, World \u0026 O\u0027Reilly\u0021`,
				`greeting=H%69,\u0026addressee=(World)`,
				`greeting=H%69,\u0026addressee=(World) 2x, https:\/\/golang.org\/favicon.ico 500.5w`,
				`,foo\/,`,
			},
		},
		{
			`<a href="?q={{.}}">`,
			[]string{
				`%3cb%3e%20%22foo%25%22%20O%27Reilly%20%26bar%3b`,
				`a%5bhref%20%3d~%20%22%2f%2fexample.com%22%5d%23foo`,
				`Hello%2c%20%3cb%3eWorld%3c%2fb%3e%20%26amp%3btc%21`,
				`%20dir%3d%22ltr%22`,
				`c%20%26%26%20alert%28%22Hello%2c%20World%21%22%29%3b`,
				`Hello%2c%20World%20%26%20O%27Reilly%5cu0021`,
				// Quotes and parens are escaped but %69 is not over-escaped. HTML escaping is done.
				`greeting=H%69,&amp;addressee=%28World%29`,
				`greeting%3dH%2569%2c%26addressee%3d%28World%29%202x%2c%20https%3a%2f%2fgolang.org%2ffavicon.ico%20500.5w`,
				`,foo/,`,
			},
		},
		{
			`<style>body { background: url('?img={{.}}') }</style>`,
			[]string{
				`%3cb%3e%20%22foo%25%22%20O%27Reilly%20%26bar%3b`,
				`a%5bhref%20%3d~%20%22%2f%2fexample.com%22%5d%23foo`,
				`Hello%2c%20%3cb%3eWorld%3c%2fb%3e%20%26amp%3btc%21`,
				`%20dir%3d%22ltr%22`,
				`c%20%26%26%20alert%28%22Hello%2c%20World%21%22%29%3b`,
				`Hello%2c%20World%20%26%20O%27Reilly%5cu0021`,
				// Quotes and parens are escaped but %69 is not over-escaped. HTML escaping is not done.
				`greeting=H%69,&addressee=%28World%29`,
				`greeting%3dH%2569%2c%26addressee%3d%28World%29%202x%2c%20https%3a%2f%2fgolang.org%2ffavicon.ico%20500.5w`,
				`,foo/,`,
			},
		},
		{
			`<img srcset="{{.}}">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				// Commas are not escaped.
				`Hello,#ZgotmplZ`,
				// Leading spaces are not percent escapes.
				` dir=%22ltr%22`,
				// Spaces after commas are not percent escaped.
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				// Metadata is not escaped.
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset={{.}}>`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				// Spaces are HTML escaped not %-escaped
				`&#32;dir&#61;%22ltr%22`,
				`#ZgotmplZ,&#32;World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting&#61;H%69%2c&amp;addressee&#61;%28World%29`,
				`greeting&#61;H%69,&amp;addressee&#61;(World)&#32;2x,&#32;https://golang.org/favicon.ico&#32;500.5w`,
				// Commas are escaped.
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset="{{.}} 2x, https://golang.org/ 500.5w">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				` dir=%22ltr%22`,
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset="http://godoc.org/ {{.}}, https://golang.org/ 500.5w">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				` dir=%22ltr%22`,
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset="http://godoc.org/?q={{.}} 2x, https://golang.org/ 500.5w">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				` dir=%22ltr%22`,
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset="http://godoc.org/ 2x, {{.}} 500.5w">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				` dir=%22ltr%22`,
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
		{
			`<img srcset="http://godoc.org/ 2x, https://golang.org/ {{.}}">`,
			[]string{
				`#ZgotmplZ`,
				`#ZgotmplZ`,
				`Hello,#ZgotmplZ`,
				` dir=%22ltr%22`,
				`#ZgotmplZ, World!%22%29;`,
				`Hello,#ZgotmplZ`,
				`greeting=H%69%2c&amp;addressee=%28World%29`,
				`greeting=H%69,&amp;addressee=(World) 2x, https://golang.org/favicon.ico 500.5w`,
				`%2cfoo/%2c`,
			},
		},
	}

	for _, test := range tests {
		tmpl := Must(New("x").Parse(test.input))
		pre := strings.Index(test.input, "{{.}}")
		post := len(test.input) - (pre + 5)
		var b strings.Builder
		for i, x := range data {
			b.Reset()
			if err := tmpl.Execute(&b, x); err != nil {
				t.Errorf("%q with %v: %s", test.input, x, err)
				continue
			}
			if want, got := test.want[i], b.String()[pre:b.Len()-post]; want != got {
				t.Errorf("%q with %v:\nwant\n\t%q,\ngot\n\t%q\n", test.input, x, want, got)
				continue
			}
		}
	}
}

// Test that we print using the String method. Was issue 3073.
type myStringer struct {
	v int
}

func (s *myStringer) String() string {
	return fmt.Sprintf("string=%d", s.v)
}

type errorer struct {
	v int
}

func (s *errorer) Error() string {
	return fmt.Sprintf("error=%d", s.v)
}

func TestStringer(t *testing.T) {
	s := &myStringer{3}
	b := new(strings.Builder)
	tmpl := Must(New("x").Parse("{{.}}"))
	if err := tmpl.Execute(b, s); err != nil {
		t.Fatal(err)
	}
	var expect = "string=3"
	if b.String() != expect {
		t.Errorf("expected %q got %q", expect, b.String())
	}
	e := &errorer{7}
	b.Reset()
	if err := tmpl.Execute(b, e); err != nil {
		t.Fatal(err)
	}
	expect = "error=7"
	if b.String() != expect {
		t.Errorf("expected %q got %q", expect, b.String())
	}
}

// https://golang.org/issue/5982
func TestEscapingNilNonemptyInterfaces(t *testing.T) {
	tmpl := Must(New("x").Parse("{{.E}}"))

	got := new(bytes.Buffer)
	testData := struct{ E error }{} // any non-empty interface here will do; error is just ready at hand
	tmpl.Execute(got, testData)

	// A non-empty interface should print like an empty interface.
	want := new(bytes.Buffer)
	data := struct{ E any }{}
	tmpl.Execute(want, data)

	if !bytes.Equal(want.Bytes(), got.Bytes()) {
		t.Errorf("expected %q got %q", string(want.Bytes()), string(got.Bytes()))
	}
}
