// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"text/template"
	"text/template/parse"
)

type badMarshaler struct{}

func (x *badMarshaler) MarshalJSON() ([]byte, error) {
	// Keys in valid JSON must be double quoted as must all strings.
	return []byte("{ foo: 'not quite valid JSON' }"), nil
}

type goodMarshaler struct{}

func (x *goodMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(`{ "<foo>": "O'Reilly" }`), nil
}

func TestEscape(t *testing.T) {
	data := struct {
		F, T    bool
		C, G, H string
		A, E    []string
		B, M    json.Marshaler
		N       int
		Z       *int
		W       HTML
	}{
		F: false,
		T: true,
		C: "<Cincinatti>",
		G: "<Goodbye>",
		H: "<Hello>",
		A: []string{"<a>", "<b>"},
		E: []string{},
		N: 42,
		B: &badMarshaler{},
		M: &goodMarshaler{},
		Z: nil,
		W: HTML(`&iexcl;<b class="foo">Hello</b>, <textarea>O'World</textarea>!`),
	}
	pdata := &data

	tests := []struct {
		name   string
		input  string
		output string
	}{
		{
			"if",
			"{{if .T}}Hello{{end}}, {{.C}}!",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"else",
			"{{if .F}}{{.H}}{{else}}{{.G}}{{end}}!",
			"&lt;Goodbye&gt;!",
		},
		{
			"overescaping1",
			"Hello, {{.C | html}}!",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"overescaping2",
			"Hello, {{html .C}}!",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"overescaping3",
			"{{with .C}}{{$msg := .}}Hello, {{$msg}}!{{end}}",
			"Hello, &lt;Cincinatti&gt;!",
		},
		{
			"assignment",
			"{{if $x := .H}}{{$x}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"withBody",
			"{{with .H}}{{.}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"withElse",
			"{{with .E}}{{.}}{{else}}{{.H}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"rangeBody",
			"{{range .A}}{{.}}{{end}}",
			"&lt;a&gt;&lt;b&gt;",
		},
		{
			"rangeElse",
			"{{range .E}}{{.}}{{else}}{{.H}}{{end}}",
			"&lt;Hello&gt;",
		},
		{
			"nonStringValue",
			"{{.T}}",
			"true",
		},
		{
			"constant",
			`<a href="/search?q={{"'a<b'"}}">`,
			`<a href="/search?q=%27a%3cb%27">`,
		},
		{
			"multipleAttrs",
			"<a b=1 c={{.H}}>",
			"<a b=1 c=&lt;Hello&gt;>",
		},
		{
			"urlStartRel",
			`<a href='{{"/foo/bar?a=b&c=d"}}'>`,
			`<a href='/foo/bar?a=b&amp;c=d'>`,
		},
		{
			"urlStartAbsOk",
			`<a href='{{"http://example.com/foo/bar?a=b&c=d"}}'>`,
			`<a href='http://example.com/foo/bar?a=b&amp;c=d'>`,
		},
		{
			"protocolRelativeURLStart",
			`<a href='{{"//example.com:8000/foo/bar?a=b&c=d"}}'>`,
			`<a href='//example.com:8000/foo/bar?a=b&amp;c=d'>`,
		},
		{
			"pathRelativeURLStart",
			`<a href="{{"/javascript:80/foo/bar"}}">`,
			`<a href="/javascript:80/foo/bar">`,
		},
		{
			"dangerousURLStart",
			`<a href='{{"javascript:alert(%22pwned%22)"}}'>`,
			`<a href='#ZgotmplZ'>`,
		},
		{
			"dangerousURLStart2",
			`<a href='  {{"javascript:alert(%22pwned%22)"}}'>`,
			`<a href='  #ZgotmplZ'>`,
		},
		{
			"nonHierURL",
			`<a href={{"mailto:Muhammed \"The Greatest\" Ali <m.ali@example.com>"}}>`,
			`<a href=mailto:Muhammed%20%22The%20Greatest%22%20Ali%20%3cm.ali@example.com%3e>`,
		},
		{
			"urlPath",
			`<a href='http://{{"javascript:80"}}/foo'>`,
			`<a href='http://javascript:80/foo'>`,
		},
		{
			"urlQuery",
			`<a href='/search?q={{.H}}'>`,
			`<a href='/search?q=%3cHello%3e'>`,
		},
		{
			"urlFragment",
			`<a href='/faq#{{.H}}'>`,
			`<a href='/faq#%3cHello%3e'>`,
		},
		{
			"urlBranch",
			`<a href="{{if .F}}/foo?a=b{{else}}/bar{{end}}">`,
			`<a href="/bar">`,
		},
		{
			"urlBranchConflictMoot",
			`<a href="{{if .T}}/foo?a={{else}}/bar#{{end}}{{.C}}">`,
			`<a href="/foo?a=%3cCincinatti%3e">`,
		},
		{
			"jsStrValue",
			"<button onclick='alert({{.H}})'>",
			`<button onclick='alert(&#34;\u003cHello\u003e&#34;)'>`,
		},
		{
			"jsNumericValue",
			"<button onclick='alert({{.N}})'>",
			`<button onclick='alert( 42 )'>`,
		},
		{
			"jsBoolValue",
			"<button onclick='alert({{.T}})'>",
			`<button onclick='alert( true )'>`,
		},
		{
			"jsNilValue",
			"<button onclick='alert(typeof{{.Z}})'>",
			`<button onclick='alert(typeof null )'>`,
		},
		{
			"jsObjValue",
			"<button onclick='alert({{.A}})'>",
			`<button onclick='alert([&#34;\u003ca\u003e&#34;,&#34;\u003cb\u003e&#34;])'>`,
		},
		{
			"jsObjValueScript",
			"<script>alert({{.A}})</script>",
			`<script>alert(["\u003ca\u003e","\u003cb\u003e"])</script>`,
		},
		{
			"jsObjValueNotOverEscaped",
			"<button onclick='alert({{.A | html}})'>",
			`<button onclick='alert([&#34;\u003ca\u003e&#34;,&#34;\u003cb\u003e&#34;])'>`,
		},
		{
			"jsStr",
			"<button onclick='alert(&quot;{{.H}}&quot;)'>",
			`<button onclick='alert(&quot;\x3cHello\x3e&quot;)'>`,
		},
		{
			"badMarshaler",
			`<button onclick='alert(1/{{.B}}in numbers)'>`,
			`<button onclick='alert(1/ /* json: error calling MarshalJSON for type *template.badMarshaler: invalid character &#39;f&#39; looking for beginning of object key string */null in numbers)'>`,
		},
		{
			"jsMarshaler",
			`<button onclick='alert({{.M}})'>`,
			`<button onclick='alert({&#34;\u003cfoo\u003e&#34;:&#34;O&#39;Reilly&#34;})'>`,
		},
		{
			"jsStrNotUnderEscaped",
			"<button onclick='alert({{.C | urlquery}})'>",
			// URL escaped, then quoted for JS.
			`<button onclick='alert(&#34;%3CCincinatti%3E&#34;)'>`,
		},
		{
			"jsRe",
			`<button onclick='alert(/{{"foo+bar"}}/.test(""))'>`,
			`<button onclick='alert(/foo\x2bbar/.test(""))'>`,
		},
		{
			"jsReBlank",
			`<script>alert(/{{""}}/.test(""));</script>`,
			`<script>alert(/(?:)/.test(""));</script>`,
		},
		{
			"jsReAmbigOk",
			`<script>{{if true}}var x = 1{{end}}</script>`,
			// The {if} ends in an ambiguous jsCtx but there is
			// no slash following so we shouldn't care.
			`<script>var x = 1</script>`,
		},
		{
			"styleBidiKeywordPassed",
			`<p style="dir: {{"ltr"}}">`,
			`<p style="dir: ltr">`,
		},
		{
			"styleBidiPropNamePassed",
			`<p style="border-{{"left"}}: 0; border-{{"right"}}: 1in">`,
			`<p style="border-left: 0; border-right: 1in">`,
		},
		{
			"styleExpressionBlocked",
			`<p style="width: {{"expression(alert(1337))"}}">`,
			`<p style="width: ZgotmplZ">`,
		},
		{
			"styleTagSelectorPassed",
			`<style>{{"p"}} { color: pink }</style>`,
			`<style>p { color: pink }</style>`,
		},
		{
			"styleIDPassed",
			`<style>p{{"#my-ID"}} { font: Arial }</style>`,
			`<style>p#my-ID { font: Arial }</style>`,
		},
		{
			"styleClassPassed",
			`<style>p{{".my_class"}} { font: Arial }</style>`,
			`<style>p.my_class { font: Arial }</style>`,
		},
		{
			"styleQuantityPassed",
			`<a style="left: {{"2em"}}; top: {{0}}">`,
			`<a style="left: 2em; top: 0">`,
		},
		{
			"stylePctPassed",
			`<table style=width:{{"100%"}}>`,
			`<table style=width:100%>`,
		},
		{
			"styleColorPassed",
			`<p style="color: {{"#8ff"}}; background: {{"#000"}}">`,
			`<p style="color: #8ff; background: #000">`,
		},
		{
			"styleObfuscatedExpressionBlocked",
			`<p style="width: {{"  e\\78preS\x00Sio/**/n(alert(1337))"}}">`,
			`<p style="width: ZgotmplZ">`,
		},
		{
			"styleMozBindingBlocked",
			`<p style="{{"-moz-binding(alert(1337))"}}: ...">`,
			`<p style="ZgotmplZ: ...">`,
		},
		{
			"styleObfuscatedMozBindingBlocked",
			`<p style="{{"  -mo\\7a-B\x00I/**/nding(alert(1337))"}}: ...">`,
			`<p style="ZgotmplZ: ...">`,
		},
		{
			"styleFontNameString",
			`<p style='font-family: "{{"Times New Roman"}}"'>`,
			`<p style='font-family: "Times New Roman"'>`,
		},
		{
			"styleFontNameString",
			`<p style='font-family: "{{"Times New Roman"}}", "{{"sans-serif"}}"'>`,
			`<p style='font-family: "Times New Roman", "sans-serif"'>`,
		},
		{
			"styleFontNameUnquoted",
			`<p style='font-family: {{"Times New Roman"}}'>`,
			`<p style='font-family: Times New Roman'>`,
		},
		{
			"styleURLQueryEncoded",
			`<p style="background: url(/img?name={{"O'Reilly Animal(1)<2>.png"}})">`,
			`<p style="background: url(/img?name=O%27Reilly%20Animal%281%29%3c2%3e.png)">`,
		},
		{
			"styleQuotedURLQueryEncoded",
			`<p style="background: url('/img?name={{"O'Reilly Animal(1)<2>.png"}}')">`,
			`<p style="background: url('/img?name=O%27Reilly%20Animal%281%29%3c2%3e.png')">`,
		},
		{
			"styleStrQueryEncoded",
			`<p style="background: '/img?name={{"O'Reilly Animal(1)<2>.png"}}'">`,
			`<p style="background: '/img?name=O%27Reilly%20Animal%281%29%3c2%3e.png'">`,
		},
		{
			"styleURLBadProtocolBlocked",
			`<a style="background: url('{{"javascript:alert(1337)"}}')">`,
			`<a style="background: url('#ZgotmplZ')">`,
		},
		{
			"styleStrBadProtocolBlocked",
			`<a style="background: '{{"vbscript:alert(1337)"}}'">`,
			`<a style="background: '#ZgotmplZ'">`,
		},
		{
			"styleStrEncodedProtocolEncoded",
			`<a style="background: '{{"javascript\\3a alert(1337)"}}'">`,
			// The CSS string 'javascript\\3a alert(1337)' does not contains a colon.
			`<a style="background: 'javascript\\3a alert\28 1337\29 '">`,
		},
		{
			"styleURLGoodProtocolPassed",
			`<a style="background: url('{{"http://oreilly.com/O'Reilly Animals(1)<2>;{}.html"}}')">`,
			`<a style="background: url('http://oreilly.com/O%27Reilly%20Animals%281%29%3c2%3e;%7b%7d.html')">`,
		},
		{
			"styleStrGoodProtocolPassed",
			`<a style="background: '{{"http://oreilly.com/O'Reilly Animals(1)<2>;{}.html"}}'">`,
			`<a style="background: 'http\3a\2f\2foreilly.com\2fO\27Reilly Animals\28 1\29\3c 2\3e\3b\7b\7d.html'">`,
		},
		{
			"styleURLEncodedForHTMLInAttr",
			`<a style="background: url('{{"/search?img=foo&size=icon"}}')">`,
			`<a style="background: url('/search?img=foo&amp;size=icon')">`,
		},
		{
			"styleURLNotEncodedForHTMLInCdata",
			`<style>body { background: url('{{"/search?img=foo&size=icon"}}') }</style>`,
			`<style>body { background: url('/search?img=foo&size=icon') }</style>`,
		},
		{
			"styleURLMixedCase",
			`<p style="background: URL(#{{.H}})">`,
			`<p style="background: URL(#%3cHello%3e)">`,
		},
		{
			"stylePropertyPairPassed",
			`<a style='{{"color: red"}}'>`,
			`<a style='color: red'>`,
		},
		{
			"styleStrSpecialsEncoded",
			`<a style="font-family: '{{"/**/'\";:// \\"}}', &quot;{{"/**/'\";:// \\"}}&quot;">`,
			`<a style="font-family: '\2f**\2f\27\22\3b\3a\2f\2f  \\', &quot;\2f**\2f\27\22\3b\3a\2f\2f  \\&quot;">`,
		},
		{
			"styleURLSpecialsEncoded",
			`<a style="border-image: url({{"/**/'\";:// \\"}}), url(&quot;{{"/**/'\";:// \\"}}&quot;), url('{{"/**/'\";:// \\"}}'), 'http://www.example.com/?q={{"/**/'\";:// \\"}}''">`,
			`<a style="border-image: url(/**/%27%22;://%20%5c), url(&quot;/**/%27%22;://%20%5c&quot;), url('/**/%27%22;://%20%5c'), 'http://www.example.com/?q=%2f%2a%2a%2f%27%22%3b%3a%2f%2f%20%5c''">`,
		},
		{
			"HTML comment",
			"<b>Hello, <!-- name of world -->{{.C}}</b>",
			"<b>Hello, &lt;Cincinatti&gt;</b>",
		},
		{
			"HTML comment not first < in text node.",
			"<<!-- -->!--",
			"&lt;!--",
		},
		{
			"HTML normalization 1",
			"a < b",
			"a &lt; b",
		},
		{
			"HTML normalization 2",
			"a << b",
			"a &lt;&lt; b",
		},
		{
			"HTML normalization 3",
			"a<<!-- --><!-- -->b",
			"a&lt;b",
		},
		{
			"HTML doctype not normalized",
			"<!DOCTYPE html>Hello, World!",
			"<!DOCTYPE html>Hello, World!",
		},
		{
			"HTML doctype not case-insensitive",
			"<!doCtYPE htMl>Hello, World!",
			"<!doCtYPE htMl>Hello, World!",
		},
		{
			"No doctype injection",
			`<!{{"DOCTYPE"}}`,
			"&lt;!DOCTYPE",
		},
		{
			"Split HTML comment",
			"<b>Hello, <!-- name of {{if .T}}city -->{{.C}}{{else}}world -->{{.W}}{{end}}</b>",
			"<b>Hello, &lt;Cincinatti&gt;</b>",
		},
		{
			"JS line comment",
			"<script>for (;;) { if (c()) break// foo not a label\n" +
				"foo({{.T}});}</script>",
			"<script>for (;;) { if (c()) break\n" +
				"foo( true );}</script>",
		},
		{
			"JS multiline block comment",
			"<script>for (;;) { if (c()) break/* foo not a label\n" +
				" */foo({{.T}});}</script>",
			// Newline separates break from call. If newline
			// removed, then break will consume label leaving
			// code invalid.
			"<script>for (;;) { if (c()) break\n" +
				"foo( true );}</script>",
		},
		{
			"JS single-line block comment",
			"<script>for (;;) {\n" +
				"if (c()) break/* foo a label */foo;" +
				"x({{.T}});}</script>",
			// Newline separates break from call. If newline
			// removed, then break will consume label leaving
			// code invalid.
			"<script>for (;;) {\n" +
				"if (c()) break foo;" +
				"x( true );}</script>",
		},
		{
			"JS block comment flush with mathematical division",
			"<script>var a/*b*//c\nd</script>",
			"<script>var a /c\nd</script>",
		},
		{
			"JS mixed comments",
			"<script>var a/*b*///c\nd</script>",
			"<script>var a \nd</script>",
		},
		{
			"CSS comments",
			"<style>p// paragraph\n" +
				`{border: 1px/* color */{{"#00f"}}}</style>`,
			"<style>p\n" +
				"{border: 1px #00f}</style>",
		},
		{
			"JS attr block comment",
			`<a onclick="f(&quot;&quot;); /* alert({{.H}}) */">`,
			// Attribute comment tests should pass if the comments
			// are successfully elided.
			`<a onclick="f(&quot;&quot;); /* alert() */">`,
		},
		{
			"JS attr line comment",
			`<a onclick="// alert({{.G}})">`,
			`<a onclick="// alert()">`,
		},
		{
			"CSS attr block comment",
			`<a style="/* color: {{.H}} */">`,
			`<a style="/* color:  */">`,
		},
		{
			"CSS attr line comment",
			`<a style="// color: {{.G}}">`,
			`<a style="// color: ">`,
		},
		{
			"HTML substitution commented out",
			"<p><!-- {{.H}} --></p>",
			"<p></p>",
		},
		{
			"Comment ends flush with start",
			"<!--{{.}}--><script>/*{{.}}*///{{.}}\n</script><style>/*{{.}}*///{{.}}\n</style><a onclick='/*{{.}}*///{{.}}' style='/*{{.}}*///{{.}}'>",
			"<script> \n</script><style> \n</style><a onclick='/**///' style='/**///'>",
		},
		{
			"typed HTML in text",
			`{{.W}}`,
			`&iexcl;<b class="foo">Hello</b>, <textarea>O'World</textarea>!`,
		},
		{
			"typed HTML in attribute",
			`<div title="{{.W}}">`,
			`<div title="&iexcl;Hello, O&#39;World!">`,
		},
		{
			"typed HTML in script",
			`<button onclick="alert({{.W}})">`,
			`<button onclick="alert(&#34;\u0026iexcl;\u003cb class=\&#34;foo\&#34;\u003eHello\u003c/b\u003e, \u003ctextarea\u003eO&#39;World\u003c/textarea\u003e!&#34;)">`,
		},
		{
			"typed HTML in RCDATA",
			`<textarea>{{.W}}</textarea>`,
			`<textarea>&iexcl;&lt;b class=&#34;foo&#34;&gt;Hello&lt;/b&gt;, &lt;textarea&gt;O&#39;World&lt;/textarea&gt;!</textarea>`,
		},
		{
			"range in textarea",
			"<textarea>{{range .A}}{{.}}{{end}}</textarea>",
			"<textarea>&lt;a&gt;&lt;b&gt;</textarea>",
		},
		{
			"No tag injection",
			`{{"10$"}}<{{"script src,evil.org/pwnd.js"}}...`,
			`10$&lt;script src,evil.org/pwnd.js...`,
		},
		{
			"No comment injection",
			`<{{"!--"}}`,
			`&lt;!--`,
		},
		{
			"No RCDATA end tag injection",
			`<textarea><{{"/textarea "}}...</textarea>`,
			`<textarea>&lt;/textarea ...</textarea>`,
		},
		{
			"optional attrs",
			`<img class="{{"iconClass"}}"` +
				`{{if .T}} id="{{"<iconId>"}}"{{end}}` +
				// Double quotes inside if/else.
				` src=` +
				`{{if .T}}"?{{"<iconPath>"}}"` +
				`{{else}}"images/cleardot.gif"{{end}}` +
				// Missing space before title, but it is not a
				// part of the src attribute.
				`{{if .T}}title="{{"<title>"}}"{{end}}` +
				// Quotes outside if/else.
				` alt="` +
				`{{if .T}}{{"<alt>"}}` +
				`{{else}}{{if .F}}{{"<title>"}}{{end}}` +
				`{{end}}"` +
				`>`,
			`<img class="iconClass" id="&lt;iconId&gt;" src="?%3ciconPath%3e"title="&lt;title&gt;" alt="&lt;alt&gt;">`,
		},
		{
			"conditional valueless attr name",
			`<input{{if .T}} checked{{end}} name=n>`,
			`<input checked name=n>`,
		},
		{
			"conditional dynamic valueless attr name 1",
			`<input{{if .T}} {{"checked"}}{{end}} name=n>`,
			`<input checked name=n>`,
		},
		{
			"conditional dynamic valueless attr name 2",
			`<input {{if .T}}{{"checked"}} {{end}}name=n>`,
			`<input checked name=n>`,
		},
		{
			"dynamic attribute name",
			`<img on{{"load"}}="alert({{"loaded"}})">`,
			// Treated as JS since quotes are inserted.
			`<img onload="alert(&#34;loaded&#34;)">`,
		},
		{
			"bad dynamic attribute name 1",
			// Allow checked, selected, disabled, but not JS or
			// CSS attributes.
			`<input {{"onchange"}}="{{"doEvil()"}}">`,
			`<input ZgotmplZ="doEvil()">`,
		},
		{
			"bad dynamic attribute name 2",
			`<div {{"sTyle"}}="{{"color: expression(alert(1337))"}}">`,
			`<div ZgotmplZ="color: expression(alert(1337))">`,
		},
		{
			"bad dynamic attribute name 3",
			// Allow title or alt, but not a URL.
			`<img {{"src"}}="{{"javascript:doEvil()"}}">`,
			`<img ZgotmplZ="javascript:doEvil()">`,
		},
		{
			"bad dynamic attribute name 4",
			// Structure preservation requires values to associate
			// with a consistent attribute.
			`<input checked {{""}}="Whose value am I?">`,
			`<input checked ZgotmplZ="Whose value am I?">`,
		},
		{
			"dynamic element name",
			`<h{{3}}><table><t{{"head"}}>...</h{{3}}>`,
			`<h3><table><thead>...</h3>`,
		},
		{
			"bad dynamic element name",
			// Dynamic element names are typically used to switch
			// between (thead, tfoot, tbody), (ul, ol), (th, td),
			// and other replaceable sets.
			// We do not currently easily support (ul, ol).
			// If we do change to support that, this test should
			// catch failures to filter out special tag names which
			// would violate the structure preservation property --
			// if any special tag name could be substituted, then
			// the content could be raw text/RCDATA for some inputs
			// and regular HTML content for others.
			`<{{"script"}}>{{"doEvil()"}}</{{"script"}}>`,
			`&lt;script>doEvil()&lt;/script>`,
		},
	}

	for _, test := range tests {
		tmpl := New(test.name)
		tmpl = Must(tmpl.Parse(test.input))
		// Check for bug 6459: Tree field was not set in Parse.
		if tmpl.Tree != tmpl.text.Tree {
			t.Errorf("%s: tree not set properly", test.name)
			continue
		}
		b := new(bytes.Buffer)
		if err := tmpl.Execute(b, data); err != nil {
			t.Errorf("%s: template execution failed: %s", test.name, err)
			continue
		}
		if w, g := test.output, b.String(); w != g {
			t.Errorf("%s: escaped output: want\n\t%q\ngot\n\t%q", test.name, w, g)
			continue
		}
		b.Reset()
		if err := tmpl.Execute(b, pdata); err != nil {
			t.Errorf("%s: template execution failed for pointer: %s", test.name, err)
			continue
		}
		if w, g := test.output, b.String(); w != g {
			t.Errorf("%s: escaped output for pointer: want\n\t%q\ngot\n\t%q", test.name, w, g)
			continue
		}
		if tmpl.Tree != tmpl.text.Tree {
			t.Errorf("%s: tree mismatch", test.name)
			continue
		}
	}
}

func TestEscapeSet(t *testing.T) {
	type dataItem struct {
		Children []*dataItem
		X        string
	}

	data := dataItem{
		Children: []*dataItem{
			{X: "foo"},
			{X: "<bar>"},
			{
				Children: []*dataItem{
					{X: "baz"},
				},
			},
		},
	}

	tests := []struct {
		inputs map[string]string
		want   string
	}{
		// The trivial set.
		{
			map[string]string{
				"main": ``,
			},
			``,
		},
		// A template called in the start context.
		{
			map[string]string{
				"main": `Hello, {{template "helper"}}!`,
				// Not a valid top level HTML template.
				// "<b" is not a full tag.
				"helper": `{{"<World>"}}`,
			},
			`Hello, &lt;World&gt;!`,
		},
		// A template called in a context other than the start.
		{
			map[string]string{
				"main": `<a onclick='a = {{template "helper"}};'>`,
				// Not a valid top level HTML template.
				// "<b" is not a full tag.
				"helper": `{{"<a>"}}<b`,
			},
			`<a onclick='a = &#34;\u003ca\u003e&#34;<b;'>`,
		},
		// A recursive template that ends in its start context.
		{
			map[string]string{
				"main": `{{range .Children}}{{template "main" .}}{{else}}{{.X}} {{end}}`,
			},
			`foo &lt;bar&gt; baz `,
		},
		// A recursive helper template that ends in its start context.
		{
			map[string]string{
				"main":   `{{template "helper" .}}`,
				"helper": `{{if .Children}}<ul>{{range .Children}}<li>{{template "main" .}}</li>{{end}}</ul>{{else}}{{.X}}{{end}}`,
			},
			`<ul><li>foo</li><li>&lt;bar&gt;</li><li><ul><li>baz</li></ul></li></ul>`,
		},
		// Co-recursive templates that end in its start context.
		{
			map[string]string{
				"main":   `<blockquote>{{range .Children}}{{template "helper" .}}{{end}}</blockquote>`,
				"helper": `{{if .Children}}{{template "main" .}}{{else}}{{.X}}<br>{{end}}`,
			},
			`<blockquote>foo<br>&lt;bar&gt;<br><blockquote>baz<br></blockquote></blockquote>`,
		},
		// A template that is called in two different contexts.
		{
			map[string]string{
				"main":   `<button onclick="title='{{template "helper"}}'; ...">{{template "helper"}}</button>`,
				"helper": `{{11}} of {{"<100>"}}`,
			},
			`<button onclick="title='11 of \x3c100\x3e'; ...">11 of &lt;100&gt;</button>`,
		},
		// A non-recursive template that ends in a different context.
		// helper starts in jsCtxRegexp and ends in jsCtxDivOp.
		{
			map[string]string{
				"main":   `<script>var x={{template "helper"}}/{{"42"}};</script>`,
				"helper": "{{126}}",
			},
			`<script>var x= 126 /"42";</script>`,
		},
		// A recursive template that ends in a similar context.
		{
			map[string]string{
				"main":      `<script>var x=[{{template "countdown" 4}}];</script>`,
				"countdown": `{{.}}{{if .}},{{template "countdown" . | pred}}{{end}}`,
			},
			`<script>var x=[ 4 , 3 , 2 , 1 , 0 ];</script>`,
		},
		// A recursive template that ends in a different context.
		/*
			{
				map[string]string{
					"main":   `<a href="/foo{{template "helper" .}}">`,
					"helper": `{{if .Children}}{{range .Children}}{{template "helper" .}}{{end}}{{else}}?x={{.X}}{{end}}`,
				},
				`<a href="/foo?x=foo?x=%3cbar%3e?x=baz">`,
			},
		*/
	}

	// pred is a template function that returns the predecessor of a
	// natural number for testing recursive templates.
	fns := FuncMap{"pred": func(a ...interface{}) (interface{}, error) {
		if len(a) == 1 {
			if i, _ := a[0].(int); i > 0 {
				return i - 1, nil
			}
		}
		return nil, fmt.Errorf("undefined pred(%v)", a)
	}}

	for _, test := range tests {
		source := ""
		for name, body := range test.inputs {
			source += fmt.Sprintf("{{define %q}}%s{{end}} ", name, body)
		}
		tmpl, err := New("root").Funcs(fns).Parse(source)
		if err != nil {
			t.Errorf("error parsing %q: %v", source, err)
			continue
		}
		var b bytes.Buffer

		if err := tmpl.ExecuteTemplate(&b, "main", data); err != nil {
			t.Errorf("%q executing %v", err.Error(), tmpl.Lookup("main"))
			continue
		}
		if got := b.String(); test.want != got {
			t.Errorf("want\n\t%q\ngot\n\t%q", test.want, got)
		}
	}

}

func TestErrors(t *testing.T) {
	tests := []struct {
		input string
		err   string
	}{
		// Non-error cases.
		{
			"{{if .Cond}}<a>{{else}}<b>{{end}}",
			"",
		},
		{
			"{{if .Cond}}<a>{{end}}",
			"",
		},
		{
			"{{if .Cond}}{{else}}<b>{{end}}",
			"",
		},
		{
			"{{with .Cond}}<div>{{end}}",
			"",
		},
		{
			"{{range .Items}}<a>{{end}}",
			"",
		},
		{
			"<a href='/foo?{{range .Items}}&{{.K}}={{.V}}{{end}}'>",
			"",
		},
		// Error cases.
		{
			"{{if .Cond}}<a{{end}}",
			"z:1: {{if}} branches",
		},
		{
			"{{if .Cond}}\n{{else}}\n<a{{end}}",
			"z:1: {{if}} branches",
		},
		{
			// Missing quote in the else branch.
			`{{if .Cond}}<a href="foo">{{else}}<a href="bar>{{end}}`,
			"z:1: {{if}} branches",
		},
		{
			// Different kind of attribute: href implies a URL.
			"<a {{if .Cond}}href='{{else}}title='{{end}}{{.X}}'>",
			"z:1: {{if}} branches",
		},
		{
			"\n{{with .X}}<a{{end}}",
			"z:2: {{with}} branches",
		},
		{
			"\n{{with .X}}<a>{{else}}<a{{end}}",
			"z:2: {{with}} branches",
		},
		{
			"{{range .Items}}<a{{end}}",
			`z:1: on range loop re-entry: "<" in attribute name: "<a"`,
		},
		{
			"\n{{range .Items}} x='<a{{end}}",
			"z:2: on range loop re-entry: {{range}} branches",
		},
		{
			"<a b=1 c={{.H}}",
			"z: ends in a non-text context: {stateAttr delimSpaceOrTagEnd",
		},
		{
			"<script>foo();",
			"z: ends in a non-text context: {stateJS",
		},
		{
			`<a href="{{if .F}}/foo?a={{else}}/bar/{{end}}{{.H}}">`,
			"z:1: {{.H}} appears in an ambiguous URL context",
		},
		{
			`<a onclick="alert('Hello \`,
			`unfinished escape sequence in JS string: "Hello \\"`,
		},
		{
			`<a onclick='alert("Hello\, World\`,
			`unfinished escape sequence in JS string: "Hello\\, World\\"`,
		},
		{
			`<a onclick='alert(/x+\`,
			`unfinished escape sequence in JS string: "x+\\"`,
		},
		{
			`<a onclick="/foo[\]/`,
			`unfinished JS regexp charset: "foo[\\]/"`,
		},
		{
			// It is ambiguous whether 1.5 should be 1\.5 or 1.5.
			// Either `var x = 1/- 1.5 /i.test(x)`
			// where `i.test(x)` is a method call of reference i,
			// or `/-1\.5/i.test(x)` which is a method call on a
			// case insensitive regular expression.
			`<script>{{if false}}var x = 1{{end}}/-{{"1.5"}}/i.test(x)</script>`,
			`'/' could start a division or regexp: "/-"`,
		},
		{
			`{{template "foo"}}`,
			"z:1: no such template \"foo\"",
		},
		{
			`<div{{template "y"}}>` +
				// Illegal starting in stateTag but not in stateText.
				`{{define "y"}} foo<b{{end}}`,
			`"<" in attribute name: " foo<b"`,
		},
		{
			`<script>reverseList = [{{template "t"}}]</script>` +
				// Missing " after recursive call.
				`{{define "t"}}{{if .Tail}}{{template "t" .Tail}}{{end}}{{.Head}}",{{end}}`,
			`: cannot compute output context for template t$htmltemplate_stateJS_elementScript`,
		},
		{
			`<input type=button value=onclick=>`,
			`html/template:z: "=" in unquoted attr: "onclick="`,
		},
		{
			`<input type=button value= onclick=>`,
			`html/template:z: "=" in unquoted attr: "onclick="`,
		},
		{
			`<input type=button value= 1+1=2>`,
			`html/template:z: "=" in unquoted attr: "1+1=2"`,
		},
		{
			"<a class=`foo>",
			"html/template:z: \"`\" in unquoted attr: \"`foo\"",
		},
		{
			`<a style=font:'Arial'>`,
			`html/template:z: "'" in unquoted attr: "font:'Arial'"`,
		},
		{
			`<a=foo>`,
			`: expected space, attr name, or end of tag, but got "=foo>"`,
		},
	}

	for _, test := range tests {
		buf := new(bytes.Buffer)
		tmpl, err := New("z").Parse(test.input)
		if err != nil {
			t.Errorf("input=%q: unexpected parse error %s\n", test.input, err)
			continue
		}
		err = tmpl.Execute(buf, nil)
		var got string
		if err != nil {
			got = err.Error()
		}
		if test.err == "" {
			if got != "" {
				t.Errorf("input=%q: unexpected error %q", test.input, got)
			}
			continue
		}
		if strings.Index(got, test.err) == -1 {
			t.Errorf("input=%q: error\n\t%q\ndoes not contain expected string\n\t%q", test.input, got, test.err)
			continue
		}
	}
}

func TestEscapeText(t *testing.T) {
	tests := []struct {
		input  string
		output context
	}{
		{
			``,
			context{},
		},
		{
			`Hello, World!`,
			context{},
		},
		{
			// An orphaned "<" is OK.
			`I <3 Ponies!`,
			context{},
		},
		{
			`<a`,
			context{state: stateTag},
		},
		{
			`<a `,
			context{state: stateTag},
		},
		{
			`<a>`,
			context{state: stateText},
		},
		{
			`<a href`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a on`,
			context{state: stateAttrName, attr: attrScript},
		},
		{
			`<a href `,
			context{state: stateAfterName, attr: attrURL},
		},
		{
			`<a style  =  `,
			context{state: stateBeforeValue, attr: attrStyle},
		},
		{
			`<a href=`,
			context{state: stateBeforeValue, attr: attrURL},
		},
		{
			`<a href=x`,
			context{state: stateURL, delim: delimSpaceOrTagEnd, urlPart: urlPartPreQuery},
		},
		{
			`<a href=x `,
			context{state: stateTag},
		},
		{
			`<a href=>`,
			context{state: stateText},
		},
		{
			`<a href=x>`,
			context{state: stateText},
		},
		{
			`<a href ='`,
			context{state: stateURL, delim: delimSingleQuote},
		},
		{
			`<a href=''`,
			context{state: stateTag},
		},
		{
			`<a href= "`,
			context{state: stateURL, delim: delimDoubleQuote},
		},
		{
			`<a href=""`,
			context{state: stateTag},
		},
		{
			`<a title="`,
			context{state: stateAttr, delim: delimDoubleQuote},
		},
		{
			`<a HREF='http:`,
			context{state: stateURL, delim: delimSingleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a Href='/`,
			context{state: stateURL, delim: delimSingleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href='"`,
			context{state: stateURL, delim: delimSingleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href="'`,
			context{state: stateURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href='&apos;`,
			context{state: stateURL, delim: delimSingleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href="&quot;`,
			context{state: stateURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href="&#34;`,
			context{state: stateURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a href=&quot;`,
			context{state: stateURL, delim: delimSpaceOrTagEnd, urlPart: urlPartPreQuery},
		},
		{
			`<img alt="1">`,
			context{state: stateText},
		},
		{
			`<img alt="1>"`,
			context{state: stateTag},
		},
		{
			`<img alt="1>">`,
			context{state: stateText},
		},
		{
			`<input checked type="checkbox"`,
			context{state: stateTag},
		},
		{
			`<a onclick="`,
			context{state: stateJS, delim: delimDoubleQuote},
		},
		{
			`<a onclick="//foo`,
			context{state: stateJSLineCmt, delim: delimDoubleQuote},
		},
		{
			"<a onclick='//\n",
			context{state: stateJS, delim: delimSingleQuote},
		},
		{
			"<a onclick='//\r\n",
			context{state: stateJS, delim: delimSingleQuote},
		},
		{
			"<a onclick='//\u2028",
			context{state: stateJS, delim: delimSingleQuote},
		},
		{
			`<a onclick="/*`,
			context{state: stateJSBlockCmt, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/*/`,
			context{state: stateJSBlockCmt, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/**/`,
			context{state: stateJS, delim: delimDoubleQuote},
		},
		{
			`<a onkeypress="&quot;`,
			context{state: stateJSDqStr, delim: delimDoubleQuote},
		},
		{
			`<a onclick='&quot;foo&quot;`,
			context{state: stateJS, delim: delimSingleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<a onclick=&#39;foo&#39;`,
			context{state: stateJS, delim: delimSpaceOrTagEnd, jsCtx: jsCtxDivOp},
		},
		{
			`<a onclick=&#39;foo`,
			context{state: stateJSSqStr, delim: delimSpaceOrTagEnd},
		},
		{
			`<a onclick="&quot;foo'`,
			context{state: stateJSDqStr, delim: delimDoubleQuote},
		},
		{
			`<a onclick="'foo&quot;`,
			context{state: stateJSSqStr, delim: delimDoubleQuote},
		},
		{
			`<A ONCLICK="'`,
			context{state: stateJSSqStr, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/`,
			context{state: stateJSRegexp, delim: delimDoubleQuote},
		},
		{
			`<a onclick="'foo'`,
			context{state: stateJS, delim: delimDoubleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<a onclick="'foo\'`,
			context{state: stateJSSqStr, delim: delimDoubleQuote},
		},
		{
			`<a onclick="'foo\'`,
			context{state: stateJSSqStr, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/foo/`,
			context{state: stateJS, delim: delimDoubleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<script>/foo/ /=`,
			context{state: stateJS, element: elementScript},
		},
		{
			`<a onclick="1 /foo`,
			context{state: stateJS, delim: delimDoubleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<a onclick="1 /*c*/ /foo`,
			context{state: stateJS, delim: delimDoubleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<a onclick="/foo[/]`,
			context{state: stateJSRegexp, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/foo\/`,
			context{state: stateJSRegexp, delim: delimDoubleQuote},
		},
		{
			`<a onclick="/foo/`,
			context{state: stateJS, delim: delimDoubleQuote, jsCtx: jsCtxDivOp},
		},
		{
			`<input checked style="`,
			context{state: stateCSS, delim: delimDoubleQuote},
		},
		{
			`<a style="//`,
			context{state: stateCSSLineCmt, delim: delimDoubleQuote},
		},
		{
			`<a style="//</script>`,
			context{state: stateCSSLineCmt, delim: delimDoubleQuote},
		},
		{
			"<a style='//\n",
			context{state: stateCSS, delim: delimSingleQuote},
		},
		{
			"<a style='//\r",
			context{state: stateCSS, delim: delimSingleQuote},
		},
		{
			`<a style="/*`,
			context{state: stateCSSBlockCmt, delim: delimDoubleQuote},
		},
		{
			`<a style="/*/`,
			context{state: stateCSSBlockCmt, delim: delimDoubleQuote},
		},
		{
			`<a style="/**/`,
			context{state: stateCSS, delim: delimDoubleQuote},
		},
		{
			`<a style="background: '`,
			context{state: stateCSSSqStr, delim: delimDoubleQuote},
		},
		{
			`<a style="background: &quot;`,
			context{state: stateCSSDqStr, delim: delimDoubleQuote},
		},
		{
			`<a style="background: '/foo?img=`,
			context{state: stateCSSSqStr, delim: delimDoubleQuote, urlPart: urlPartQueryOrFrag},
		},
		{
			`<a style="background: '/`,
			context{state: stateCSSSqStr, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url(&#x22;/`,
			context{state: stateCSSDqURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url('/`,
			context{state: stateCSSSqURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url('/)`,
			context{state: stateCSSSqURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url('/ `,
			context{state: stateCSSSqURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url(/`,
			context{state: stateCSSURL, delim: delimDoubleQuote, urlPart: urlPartPreQuery},
		},
		{
			`<a style="background: url( `,
			context{state: stateCSSURL, delim: delimDoubleQuote},
		},
		{
			`<a style="background: url( /image?name=`,
			context{state: stateCSSURL, delim: delimDoubleQuote, urlPart: urlPartQueryOrFrag},
		},
		{
			`<a style="background: url(x)`,
			context{state: stateCSS, delim: delimDoubleQuote},
		},
		{
			`<a style="background: url('x'`,
			context{state: stateCSS, delim: delimDoubleQuote},
		},
		{
			`<a style="background: url( x `,
			context{state: stateCSS, delim: delimDoubleQuote},
		},
		{
			`<!-- foo`,
			context{state: stateHTMLCmt},
		},
		{
			`<!-->`,
			context{state: stateHTMLCmt},
		},
		{
			`<!--->`,
			context{state: stateHTMLCmt},
		},
		{
			`<!-- foo -->`,
			context{state: stateText},
		},
		{
			`<script`,
			context{state: stateTag, element: elementScript},
		},
		{
			`<script `,
			context{state: stateTag, element: elementScript},
		},
		{
			`<script src="foo.js" `,
			context{state: stateTag, element: elementScript},
		},
		{
			`<script src='foo.js' `,
			context{state: stateTag, element: elementScript},
		},
		{
			`<script type=text/javascript `,
			context{state: stateTag, element: elementScript},
		},
		{
			`<script>foo`,
			context{state: stateJS, jsCtx: jsCtxDivOp, element: elementScript},
		},
		{
			`<script>foo</script>`,
			context{state: stateText},
		},
		{
			`<script>foo</script><!--`,
			context{state: stateHTMLCmt},
		},
		{
			`<script>document.write("<p>foo</p>");`,
			context{state: stateJS, element: elementScript},
		},
		{
			`<script>document.write("<p>foo<\/script>");`,
			context{state: stateJS, element: elementScript},
		},
		{
			`<script>document.write("<script>alert(1)</script>");`,
			context{state: stateText},
		},
		{
			`<Script>`,
			context{state: stateJS, element: elementScript},
		},
		{
			`<SCRIPT>foo`,
			context{state: stateJS, jsCtx: jsCtxDivOp, element: elementScript},
		},
		{
			`<textarea>value`,
			context{state: stateRCDATA, element: elementTextarea},
		},
		{
			`<textarea>value</TEXTAREA>`,
			context{state: stateText},
		},
		{
			`<textarea name=html><b`,
			context{state: stateRCDATA, element: elementTextarea},
		},
		{
			`<title>value`,
			context{state: stateRCDATA, element: elementTitle},
		},
		{
			`<style>value`,
			context{state: stateCSS, element: elementStyle},
		},
		{
			`<a xlink:href`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a xmlns`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a xmlns:foo`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a xmlnsxyz`,
			context{state: stateAttrName},
		},
		{
			`<a data-url`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a data-iconUri`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a data-urlItem`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a g:`,
			context{state: stateAttrName},
		},
		{
			`<a g:url`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a g:iconUri`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a g:urlItem`,
			context{state: stateAttrName, attr: attrURL},
		},
		{
			`<a g:value`,
			context{state: stateAttrName},
		},
		{
			`<a svg:style='`,
			context{state: stateCSS, delim: delimSingleQuote},
		},
		{
			`<svg:font-face`,
			context{state: stateTag},
		},
		{
			`<svg:a svg:onclick="`,
			context{state: stateJS, delim: delimDoubleQuote},
		},
	}

	for _, test := range tests {
		b, e := []byte(test.input), newEscaper(nil)
		c := e.escapeText(context{}, &parse.TextNode{NodeType: parse.NodeText, Text: b})
		if !test.output.eq(c) {
			t.Errorf("input %q: want context\n\t%v\ngot\n\t%v", test.input, test.output, c)
			continue
		}
		if test.input != string(b) {
			t.Errorf("input %q: text node was modified: want %q got %q", test.input, test.input, b)
			continue
		}
	}
}

func TestEnsurePipelineContains(t *testing.T) {
	tests := []struct {
		input, output string
		ids           []string
	}{
		{
			"{{.X}}",
			".X",
			[]string{},
		},
		{
			"{{.X | html}}",
			".X | html",
			[]string{},
		},
		{
			"{{.X}}",
			".X | html",
			[]string{"html"},
		},
		{
			"{{.X | html}}",
			".X | html | urlquery",
			[]string{"urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			".X | html | urlquery",
			[]string{"urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			".X | html | urlquery",
			[]string{"html", "urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			".X | html | urlquery",
			[]string{"html"},
		},
		{
			"{{.X | urlquery}}",
			".X | html | urlquery",
			[]string{"html", "urlquery"},
		},
		{
			"{{.X | html | print}}",
			".X | urlquery | html | print",
			[]string{"urlquery", "html"},
		},
		{
			"{{($).X | html | print}}",
			"($).X | urlquery | html | print",
			[]string{"urlquery", "html"},
		},
	}
	for i, test := range tests {
		tmpl := template.Must(template.New("test").Parse(test.input))
		action, ok := (tmpl.Tree.Root.Nodes[0].(*parse.ActionNode))
		if !ok {
			t.Errorf("#%d: First node is not an action: %s", i, test.input)
			continue
		}
		pipe := action.Pipe
		ensurePipelineContains(pipe, test.ids)
		got := pipe.String()
		if got != test.output {
			t.Errorf("#%d: %s, %v: want\n\t%s\ngot\n\t%s", i, test.input, test.ids, test.output, got)
		}
	}
}

func TestEscapeErrorsNotIgnorable(t *testing.T) {
	var b bytes.Buffer
	tmpl, _ := New("dangerous").Parse("<a")
	err := tmpl.Execute(&b, nil)
	if err == nil {
		t.Errorf("Expected error")
	} else if b.Len() != 0 {
		t.Errorf("Emitted output despite escaping failure")
	}
}

func TestEscapeSetErrorsNotIgnorable(t *testing.T) {
	var b bytes.Buffer
	tmpl, err := New("root").Parse(`{{define "t"}}<a{{end}}`)
	if err != nil {
		t.Errorf("failed to parse set: %q", err)
	}
	err = tmpl.ExecuteTemplate(&b, "t", nil)
	if err == nil {
		t.Errorf("Expected error")
	} else if b.Len() != 0 {
		t.Errorf("Emitted output despite escaping failure")
	}
}

func TestRedundantFuncs(t *testing.T) {
	inputs := []interface{}{
		"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f" +
			"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
			` !"#$%&'()*+,-./` +
			`0123456789:;<=>?` +
			`@ABCDEFGHIJKLMNO` +
			`PQRSTUVWXYZ[\]^_` +
			"`abcdefghijklmno" +
			"pqrstuvwxyz{|}~\x7f" +
			"\u00A0\u0100\u2028\u2029\ufeff\ufdec\ufffd\uffff\U0001D11E" +
			"&amp;%22\\",
		CSS(`a[href =~ "//example.com"]#foo`),
		HTML(`Hello, <b>World</b> &amp;tc!`),
		HTMLAttr(` dir="ltr"`),
		JS(`c && alert("Hello, World!");`),
		JSStr(`Hello, World & O'Reilly\x21`),
		URL(`greeting=H%69&addressee=(World)`),
	}

	for n0, m := range redundantFuncs {
		f0 := funcMap[n0].(func(...interface{}) string)
		for n1 := range m {
			f1 := funcMap[n1].(func(...interface{}) string)
			for _, input := range inputs {
				want := f0(input)
				if got := f1(want); want != got {
					t.Errorf("%s %s with %T %q: want\n\t%q,\ngot\n\t%q", n0, n1, input, input, want, got)
				}
			}
		}
	}
}

func TestIndirectPrint(t *testing.T) {
	a := 3
	ap := &a
	b := "hello"
	bp := &b
	bpp := &bp
	tmpl := Must(New("t").Parse(`{{.}}`))
	var buf bytes.Buffer
	err := tmpl.Execute(&buf, ap)
	if err != nil {
		t.Errorf("Unexpected error: %s", err)
	} else if buf.String() != "3" {
		t.Errorf(`Expected "3"; got %q`, buf.String())
	}
	buf.Reset()
	err = tmpl.Execute(&buf, bpp)
	if err != nil {
		t.Errorf("Unexpected error: %s", err)
	} else if buf.String() != "hello" {
		t.Errorf(`Expected "hello"; got %q`, buf.String())
	}
}

// This is a test for issue 3272.
func TestEmptyTemplate(t *testing.T) {
	page := Must(New("page").ParseFiles(os.DevNull))
	if err := page.ExecuteTemplate(os.Stdout, "page", "nothing"); err == nil {
		t.Fatal("expected error")
	}
}

func BenchmarkEscapedExecute(b *testing.B) {
	tmpl := Must(New("t").Parse(`<a onclick="alert('{{.}}')">{{.}}</a>`))
	var buf bytes.Buffer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmpl.Execute(&buf, "foo & 'bar' & baz")
		buf.Reset()
	}
}
