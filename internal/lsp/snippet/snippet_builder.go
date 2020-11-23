// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package snippet implements the specification for the LSP snippet format.
//
// Snippets are "tab stop" templates returned as an optional attribute of LSP
// completion candidates. As the user presses tab, they cycle through a series of
// tab stops defined in the snippet. Each tab stop can optionally have placeholder
// text, which can be pre-selected by editors. For a full description of syntax
// and features, see "Snippet Syntax" at
// https://microsoft.github.io/language-server-protocol/specifications/specification-3-14/#textDocument_completion.
//
// A typical snippet looks like "foo(${1:i int}, ${2:s string})".
package snippet

import (
	"fmt"
	"strings"
)

// A Builder is used to build an LSP snippet piecemeal.
// The zero value is ready to use. Do not copy a non-zero Builder.
type Builder struct {
	// currentTabStop is the index of the previous tab stop. The
	// next tab stop will be currentTabStop+1.
	currentTabStop int
	sb             strings.Builder
}

// Escape characters defined in https://microsoft.github.io/language-server-protocol/specifications/specification-3-14/#textDocument_completion under "Grammar".
var replacer = strings.NewReplacer(
	`\`, `\\`,
	`}`, `\}`,
	`$`, `\$`,
)

func (b *Builder) WriteText(s string) {
	replacer.WriteString(&b.sb, s)
}

func (b *Builder) PrependText(s string) {
	rawSnip := b.String()
	b.sb.Reset()
	b.WriteText(s)
	b.sb.WriteString(rawSnip)
}

func (b *Builder) Write(data []byte) (int, error) {
	return b.sb.Write(data)
}

// WritePlaceholder writes a tab stop and placeholder value to the Builder.
// The callback style allows for creating nested placeholders. To write an
// empty tab stop, provide a nil callback.
func (b *Builder) WritePlaceholder(fn func(*Builder)) {
	fmt.Fprintf(&b.sb, "${%d:", b.nextTabStop())
	if fn != nil {
		fn(b)
	}
	b.sb.WriteByte('}')
}

// WriteFinalTabstop marks where cursor ends up after the user has
// cycled through all the normal tab stops. It defaults to the
// character after the snippet.
func (b *Builder) WriteFinalTabstop() {
	fmt.Fprint(&b.sb, "$0")
}

// In addition to '\', '}', and '$', snippet choices also use '|' and ',' as
// meta characters, so they must be escaped within the choices.
var choiceReplacer = strings.NewReplacer(
	`\`, `\\`,
	`}`, `\}`,
	`$`, `\$`,
	`|`, `\|`,
	`,`, `\,`,
)

// WriteChoice writes a tab stop and list of text choices to the Builder.
// The user's editor will prompt the user to choose one of the choices.
func (b *Builder) WriteChoice(choices []string) {
	fmt.Fprintf(&b.sb, "${%d|", b.nextTabStop())
	for i, c := range choices {
		if i != 0 {
			b.sb.WriteByte(',')
		}
		choiceReplacer.WriteString(&b.sb, c)
	}
	b.sb.WriteString("|}")
}

// String returns the built snippet string.
func (b *Builder) String() string {
	return b.sb.String()
}

// nextTabStop returns the next tab stop index for a new placeholder.
func (b *Builder) nextTabStop() int {
	// Tab stops start from 1, so increment before returning.
	b.currentTabStop++
	return b.currentTabStop
}
