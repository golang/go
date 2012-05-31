// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// This program generates table.go
// Invoke as
//
//	go run gen.go |gofmt >table.go

import (
	"fmt"
	"sort"
)

// identifier converts s to a Go exported identifier.
// It converts "div" to "Div" and "accept-charset" to "AcceptCharset".
func identifier(s string) string {
	b := make([]byte, 0, len(s))
	cap := true
	for _, c := range s {
		if c == '-' {
			cap = true
			continue
		}
		if cap && 'a' <= c && c <= 'z' {
			c -= 'a' - 'A'
		}
		cap = false
		b = append(b, byte(c))
	}
	return string(b)
}

func main() {
	m := map[string]bool{
		"": true,
	}
	for _, list := range [][]string{elements, attributes, eventHandlers, extra} {
		for _, s := range list {
			m[s] = true
		}
	}
	atoms := make([]string, 0, len(m))
	for s := range m {
		atoms = append(atoms, s)
	}
	sort.Strings(atoms)

	byInt := []string{}
	byStr := map[string]int{}
	ident := []string{}
	for i, s := range atoms {
		byInt = append(byInt, s)
		byStr[s] = i
		ident = append(ident, identifier(s))
	}

	fmt.Printf("package atom\n\nconst (\n")
	for i, _ := range byInt {
		if i == 0 {
			continue
		}
		fmt.Printf("\t%s Atom = %d\n", ident[i], i)
	}
	fmt.Printf(")\n\n")
	fmt.Printf("const max Atom = %d\n\n", len(byInt)-1)
	fmt.Printf("var table = []string{\n")
	for _, s := range byInt {
		fmt.Printf("\t%q,\n", s)
	}
	fmt.Printf("}\n\n")
	fmt.Printf("var oneByteAtoms = [26]Atom{\n")
	for i := 'a'; i <= 'z'; i++ {
		val := "0"
		if x := byStr[string(i)]; x != 0 {
			val = ident[x]
		}
		fmt.Printf("\t%s,\n", val)
	}
	fmt.Printf("}\n\n")
}

// The lists of element names and attribute keys were taken from
// http://www.whatwg.org/specs/web-apps/current-work/multipage/section-index.html
// as of the "HTML Living Standard - Last Updated 30 May 2012" version.

var elements = []string{
	"a",
	"abbr",
	"address",
	"area",
	"article",
	"aside",
	"audio",
	"b",
	"base",
	"bdi",
	"bdo",
	"blockquote",
	"body",
	"br",
	"button",
	"canvas",
	"caption",
	"cite",
	"code",
	"col",
	"colgroup",
	"command",
	"data",
	"datalist",
	"dd",
	"del",
	"details",
	"dfn",
	"dialog",
	"div",
	"dl",
	"dt",
	"em",
	"embed",
	"fieldset",
	"figcaption",
	"figure",
	"footer",
	"form",
	"h1",
	"h2",
	"h3",
	"h4",
	"h5",
	"h6",
	"head",
	"header",
	"hgroup",
	"hr",
	"html",
	"i",
	"iframe",
	"img",
	"input",
	"ins",
	"kbd",
	"keygen",
	"label",
	"legend",
	"li",
	"link",
	"map",
	"mark",
	"menu",
	"meta",
	"meter",
	"nav",
	"noscript",
	"object",
	"ol",
	"optgroup",
	"option",
	"output",
	"p",
	"param",
	"pre",
	"progress",
	"q",
	"rp",
	"rt",
	"ruby",
	"s",
	"samp",
	"script",
	"section",
	"select",
	"small",
	"source",
	"span",
	"strong",
	"style",
	"sub",
	"summary",
	"sup",
	"table",
	"tbody",
	"td",
	"textarea",
	"tfoot",
	"th",
	"thead",
	"time",
	"title",
	"tr",
	"track",
	"u",
	"ul",
	"var",
	"video",
	"wbr",
}

var attributes = []string{
	"accept",
	"accept-charset",
	"accesskey",
	"action",
	"alt",
	"async",
	"autocomplete",
	"autofocus",
	"autoplay",
	"border",
	"challenge",
	"charset",
	"checked",
	"cite",
	"class",
	"cols",
	"colspan",
	"command",
	"content",
	"contenteditable",
	"contextmenu",
	"controls",
	"coords",
	"crossorigin",
	"data",
	"datetime",
	"default",
	"defer",
	"dir",
	"dirname",
	"disabled",
	"download",
	"draggable",
	"dropzone",
	"enctype",
	"for",
	"form",
	"formaction",
	"formenctype",
	"formmethod",
	"formnovalidate",
	"formtarget",
	"headers",
	"height",
	"hidden",
	"high",
	"href",
	"hreflang",
	"http-equiv",
	"icon",
	"id",
	"inert",
	"ismap",
	"itemid",
	"itemprop",
	"itemref",
	"itemscope",
	"itemtype",
	"keytype",
	"kind",
	"label",
	"lang",
	"list",
	"loop",
	"low",
	"manifest",
	"max",
	"maxlength",
	"media",
	"mediagroup",
	"method",
	"min",
	"multiple",
	"muted",
	"name",
	"novalidate",
	"open",
	"optimum",
	"pattern",
	"ping",
	"placeholder",
	"poster",
	"preload",
	"radiogroup",
	"readonly",
	"rel",
	"required",
	"reversed",
	"rows",
	"rowspan",
	"sandbox",
	"spellcheck",
	"scope",
	"scoped",
	"seamless",
	"selected",
	"shape",
	"size",
	"sizes",
	"span",
	"src",
	"srcdoc",
	"srclang",
	"start",
	"step",
	"style",
	"tabindex",
	"target",
	"title",
	"translate",
	"type",
	"typemustmatch",
	"usemap",
	"value",
	"width",
	"wrap",
}

var eventHandlers = []string{
	"onabort",
	"onafterprint",
	"onbeforeprint",
	"onbeforeunload",
	"onblur",
	"oncancel",
	"oncanplay",
	"oncanplaythrough",
	"onchange",
	"onclick",
	"onclose",
	"oncontextmenu",
	"oncuechange",
	"ondblclick",
	"ondrag",
	"ondragend",
	"ondragenter",
	"ondragleave",
	"ondragover",
	"ondragstart",
	"ondrop",
	"ondurationchange",
	"onemptied",
	"onended",
	"onerror",
	"onfocus",
	"onhashchange",
	"oninput",
	"oninvalid",
	"onkeydown",
	"onkeypress",
	"onkeyup",
	"onload",
	"onloadeddata",
	"onloadedmetadata",
	"onloadstart",
	"onmessage",
	"onmousedown",
	"onmousemove",
	"onmouseout",
	"onmouseover",
	"onmouseup",
	"onmousewheel",
	"onoffline",
	"ononline",
	"onpagehide",
	"onpageshow",
	"onpause",
	"onplay",
	"onplaying",
	"onpopstate",
	"onprogress",
	"onratechange",
	"onreset",
	"onresize",
	"onscroll",
	"onseeked",
	"onseeking",
	"onselect",
	"onshow",
	"onstalled",
	"onstorage",
	"onsubmit",
	"onsuspend",
	"ontimeupdate",
	"onunload",
	"onvolumechange",
	"onwaiting",
}

// extra are ad-hoc values not covered by any of the lists above.
var extra = []string{
	"align",
	"annotation",
	"applet",
	"center",
	"color",
	"font",
	"frame",
	"frameset",
	"nobr",
}
