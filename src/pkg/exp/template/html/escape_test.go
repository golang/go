// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"strings"
	"template"
	"template/parse"
	"testing"
)

func TestEscape(t *testing.T) {
	var data = struct {
		F, T    bool
		C, G, H string
		A, E    []string
		N       int
		Z       *int
	}{
		F: false,
		T: true,
		C: "<Cincinatti>",
		G: "<Goodbye>",
		H: "<Hello>",
		A: []string{"<a>", "<b>"},
		E: []string{},
		N: 42,
		Z: nil,
	}

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
			"overescaping",
			"Hello, {{.C | html}}!",
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
			"nonHierURL",
			`<a href={{"mailto:Muhammed \"The Greatest\" Ali <m.ali@example.com>"}}>`,
			`<a href=mailto:Muhammed&#32;&#34;The&#32;Greatest&#34;&#32;Ali&#32;&lt;m.ali@example.com&gt;>`,
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
			`<p style="width: {{"  e\78preS\0Sio/**/n(alert(1337))"}}">`,
			`<p style="width: ZgotmplZ">`,
		},
		{
			"styleMozBindingBlocked",
			`<p style="{{"-moz-binding(alert(1337))"}}: ...">`,
			`<p style="ZgotmplZ: ...">`,
		},
		{
			"styleObfuscatedMozBindingBlocked",
			`<p style="{{"  -mo\7a-B\0I/**/nding(alert(1337))"}}: ...">`,
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
			`<a style="background: '{{"javascript:alert(1337)"}}'">`,
			`<a style="background: '#ZgotmplZ'">`,
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
			`<a style="font-family: '\2f**\2f\27\22\3b\3a\2f\2f \\', &quot;\2f**\2f\27\22\3b\3a\2f\2f \\&quot;">`,
		},
		{
			"styleURLSpecialsEncoded",
			// TODO: Find out what IE does with url(/*foo*/bar)
			// FF, Chrome, and Safari seem to treat it as a URL.
			`<a style="border-image: url({{"/**/'\";:// \\"}}), url(&quot;{{"/**/'\";:// \\"}}&quot;), url('{{"/**/'\";:// \\"}}'), 'http://www.example.com/?q={{"/**/'\";:// \\"}}''">`,
			`<a style="border-image: url(/**/%27%22;://%20%5c), url(&quot;/**/%27%22;://%20%5c&quot;), url('/**/%27%22;://%20%5c'), 'http://www.example.com/?q=%2f%2a%2a%2f%27%22%3b%3a%2f%2f%20%5c''">`,
		},
		{
			"comment",
			"<b>Hello, <!-- name of world -->{{.C}}</b>",
			// TODO: Elide comment.
			"<b>Hello, <!-- name of world -->&lt;Cincinatti&gt;</b>",
		},
	}

	for _, test := range tests {
		tmpl := template.Must(template.New(test.name).Parse(test.input))
		tmpl = template.Must(Escape(tmpl))
		b := new(bytes.Buffer)
		if err := tmpl.Execute(b, data); err != nil {
			t.Errorf("%s: template execution failed: %s", test.name, err)
			continue
		}
		if w, g := test.output, b.String(); w != g {
			t.Errorf("%s: escaped output: want\n\t%q\ngot\n\t%q", test.name, w, g)
			continue
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
			"z ends in a non-text context: {stateAttr delimSpaceOrTagEnd",
		},
		{
			"<script>foo();",
			"z ends in a non-text context: {stateJS",
		},
		{
			`<a href="{{if .F}}/foo?a={{else}}/bar/{{end}}{{.H}}">`,
			"z:1: (action: [(command: [F=[H]])]) appears in an ambiguous URL context",
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
			`unfinished escape sequence in JS regexp: "x+\\"`,
		},
		{
			`<a onclick="/foo[\]/`,
			`unfinished JS regexp charset: "foo[\\]/"`,
		},
		{
			`<a onclick="/* alert({{.X}}) */">`,
			`z:1: (action: [(command: [F=[X]])]) appears inside a comment`,
		},
		{
			`<a onclick="// alert({{.X}})">`,
			`z:1: (action: [(command: [F=[X]])]) appears inside a comment`,
		},
		{
			`<a style="/* color: {{.X}} */">`,
			`z:1: (action: [(command: [F=[X]])]) appears inside a comment`,
		},
		{
			`<a style="// color: {{.X}}">`,
			`z:1: (action: [(command: [F=[X]])]) appears inside a comment`,
		},
		{
			"<!-- {{.H}} -->",
			"z:1: (action: [(command: [F=[H]])]) appears inside a comment",
		},
		{
			// It is ambiguous whether 1.5 should be 1\.5 or 1.5.
			// Either `var x = 1/- 1.5 /i.test(x)`
			// where `i.test(x)` is a method call of reference i,
			// or `/-1\.5/i.test(x)` which is a method call on a
			// case insensitive regular expression.
			`<script>{{if false}}var x = 1{{end}}/-{{"1.5"}}/i.test(x)</script>`,
			`: '/' could start div or regexp: "/-"`,
		},
	}

	for _, test := range tests {
		tmpl := template.Must(template.New("z").Parse(test.input))
		var got string
		if _, err := Escape(tmpl); err != nil {
			got = err.String()
		}
		if test.err == "" {
			if got != "" {
				t.Errorf("input=%q: unexpected error %q", test.input, got)
			}
			continue
		}
		if strings.Index(got, test.err) == -1 {
			t.Errorf("input=%q: error %q does not contain expected string %q", test.input, got, test.err)
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
			`<a href=`,
			context{state: stateURL, delim: delimSpaceOrTagEnd},
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
			context{state: stateComment},
		},
		{
			`<!-->`,
			context{state: stateComment},
		},
		{
			`<!--->`,
			context{state: stateComment},
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
			context{state: stateComment},
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
	}

	for _, test := range tests {
		b := []byte(test.input)
		c := escapeText(context{}, b)
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
			"[(command: [F=[X]])]",
			[]string{},
		},
		{
			"{{.X | html}}",
			"[(command: [F=[X]]) (command: [I=html])]",
			[]string{},
		},
		{
			"{{.X}}",
			"[(command: [F=[X]]) (command: [I=html])]",
			[]string{"html"},
		},
		{
			"{{.X | html}}",
			"[(command: [F=[X]]) (command: [I=html]) (command: [I=urlquery])]",
			[]string{"urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			"[(command: [F=[X]]) (command: [I=html]) (command: [I=urlquery])]",
			[]string{"urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			"[(command: [F=[X]]) (command: [I=html]) (command: [I=urlquery])]",
			[]string{"html", "urlquery"},
		},
		{
			"{{.X | html | urlquery}}",
			"[(command: [F=[X]]) (command: [I=html]) (command: [I=urlquery])]",
			[]string{"html"},
		},
		{
			"{{.X | urlquery}}",
			"[(command: [F=[X]]) (command: [I=html]) (command: [I=urlquery])]",
			[]string{"html", "urlquery"},
		},
		{
			"{{.X | html | print}}",
			"[(command: [F=[X]]) (command: [I=urlquery]) (command: [I=html]) (command: [I=print])]",
			[]string{"urlquery", "html"},
		},
	}
	for _, test := range tests {
		tmpl := template.Must(template.New("test").Parse(test.input))
		action, ok := (tmpl.Tree.Root.Nodes[0].(*parse.ActionNode))
		if !ok {
			t.Errorf("First node is not an action: %s", test.input)
			continue
		}
		pipe := action.Pipe
		ensurePipelineContains(pipe, test.ids)
		got := pipe.String()
		if got != test.output {
			t.Errorf("%s, %v: want\n\t%s\ngot\n\t%s", test.input, test.ids, test.output, got)
		}
	}
}
