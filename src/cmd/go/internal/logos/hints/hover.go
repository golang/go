// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"fmt"
	"strings"
)

// siteTitle is the spelling of each site kind in a hover.
var siteTitle = map[string]string{
	SiteTry:         "?",
	SiteConstFor:    "const for",
	SiteConstSwitch: "const switch",
	SiteInterp:      `$"..."`,
}

// HoverAt returns the hover of the lowering site under pos in the file named
// file, or nil when no site lies there. It shows the lowered code and, for an
// interpolation, `Fixed: <n>B + dynamic` (SPEC Sec 21.3). When sites nest,
// the innermost answers.
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

// hoverText renders the markdown of a site: its title, the fixed byte count
// of an interpolation, and the lowered code in a fence longer than any run of
// backquotes inside it.
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
