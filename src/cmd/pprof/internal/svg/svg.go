// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package svg provides tools related to handling of SVG files
package svg

import (
	"bytes"
	"regexp"
	"strings"
)

var (
	viewBox  = regexp.MustCompile(`<svg\s*width="[^"]+"\s*height="[^"]+"\s*viewBox="[^"]+"`)
	graphId  = regexp.MustCompile(`<g id="graph\d"`)
	svgClose = regexp.MustCompile(`</svg>`)
)

// Massage enhances the SVG output from DOT to provide better
// panning inside a web browser. It uses the SVGPan library, which is
// included directly.
func Massage(in bytes.Buffer) string {
	svg := string(in.Bytes())

	// Work around for dot bug which misses quoting some ampersands,
	// resulting on unparsable SVG.
	svg = strings.Replace(svg, "&;", "&amp;;", -1)

	//Dot's SVG output is
	//
	//    <svg width="___" height="___"
	//     viewBox="___" xmlns=...>
	//    <g id="graph0" transform="...">
	//    ...
	//    </g>
	//    </svg>
	//
	// Change it to
	//
	//    <svg width="100%" height="100%"
	//     xmlns=...>
	//    <script>...</script>
	//    <g id="viewport" transform="translate(0,0)">
	//    <g id="graph0" transform="...">
	//    ...
	//    </g>
	//    </g>
	//    </svg>

	if loc := viewBox.FindStringIndex(svg); loc != nil {
		svg = svg[:loc[0]] +
			`<svg width="100%" height="100%"` +
			svg[loc[1]:]
	}

	if loc := graphId.FindStringIndex(svg); loc != nil {
		svg = svg[:loc[0]] +
			`<script type="text/ecmascript"><![CDATA[` + svgPanJS + `]]></script>` +
			`<g id="viewport" transform="scale(0.5,0.5) translate(0,0)">` +
			svg[loc[0]:]
	}

	if loc := svgClose.FindStringIndex(svg); loc != nil {
		svg = svg[:loc[0]] +
			`</g>` +
			svg[loc[0]:]
	}

	return svg
}
