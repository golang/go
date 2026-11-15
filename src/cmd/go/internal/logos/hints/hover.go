// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"fmt"
	"strings"
)

var siteTitle = map[string]string{
	SiteTry:         "?",
	SiteConstFor:    "const for",
	SiteConstSwitch: "const switch",
	SiteInterp:      `$"..."`,
}

func HoverAt(r *Report, file string, pos Position) *Hover {
	f := r.file(file)
	if f == nil {
		return nil
	}
	var best *Site
	for i := range f.Sites {
		if s := &f.Sites[i]; s.Range.holds(pos) && (best == nil || best.Range.Start.before(s.Range.Start)) {
			best = s
		}
	}
	if best == nil {
		return nil
	}
	return &Hover{
		Contents: MarkupContent{Kind: "markdown", Value: hoverText(*best)},
		Range:    &best.Range,
	}
}

func hoverText(s Site) string {
	var b strings.Builder
	fmt.Fprintf(&b, "**Lowered** `%s`\n\n", siteTitle[s.Kind])
	if s.FixedBytes != nil {
		fmt.Fprintf(&b, "Fixed: %dB + dynamic\n\n", *s.FixedBytes)
	}
	fence := "```"
	for strings.Contains(s.Lowered, fence) {
		fence += "`"
	}
	b.WriteString(fence + "go\n" + s.Lowered + "\n" + fence)
	return b.String()
}
