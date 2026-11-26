// Copyright 2026 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package dotwriter emits Graphviz DOT digraphs with automatic escaping of
// attribute values and identifiers so callers cannot inject raw DOT syntax.
package dotwriter

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Text holds a string whose contents are already escaped for inclusion inside
// a double-quoted DOT string literal (and may contain intentional DOT escape
// sequences such as \n or \l). Because raw is unexported, callers outside this
// package cannot convert an unescaped string to Text directly.
type Text struct {
	raw string
}

// String returns the DOT-escaped representation (including any \n or \l
// sequences) without surrounding quotes.
func (t Text) String() string { return t.raw }

// Escaped escapes s for DOT and wraps it as a single-segment Text.
func Escaped(s string) Text {
	return Text{escape(s)}
}

// LinesCenter escapes each line and joins them with Graphviz centered line
// breaks (\n).
func LinesCenter(lines []string) Text {
	return Text{strings.Join(escapeAll(lines), `\n`)}
}

// LinesLeft escapes each line and terminates every line with a Graphviz
// left-justified line break (\l).
func LinesLeft(lines []string) Text {
	if len(lines) == 0 {
		return Text{}
	}
	return Text{strings.Join(escapeAll(lines), `\l`) + `\l`}
}

// Join concatenates multiple already-escaped Text values without inserting
// line breaks.
func Join(parts ...Text) Text {
	var b strings.Builder
	for _, p := range parts {
		b.WriteString(p.raw)
	}
	return Text{b.String()}
}

// JoinLinesCenter joins non-empty Text values with centered line breaks (\n).
func JoinLinesCenter(lines ...Text) Text {
	var b strings.Builder
	for i, l := range lines {
		if i > 0 {
			b.WriteString(`\n`)
		}
		b.WriteString(l.raw)
	}
	return Text{b.String()}
}

// ID is a DOT identifier (emitted bare if it matches [A-Za-z_][A-Za-z0-9_]*,
// otherwise double-quoted and escaped).
type ID string

// Attr is a single key=value attribute on a DOT node, edge, or graph statement.
// Supported val types are string (auto-escaped and quoted), Text (quoted
// without re-escaping), ID (bare if safe, otherwise quoted and escaped), and
// int.
type Attr struct {
	Key string
	Val any
}

// Writer writes a single DOT digraph to an underlying io.Writer without
// exposing raw write methods to callers.
type Writer struct {
	out io.Writer
}

// Begin starts a DOT digraph with the given name and returns a Writer for
// emitting statements inside it. Call End to close the digraph.
func Begin(out io.Writer, name string) *Writer {
	w := &Writer{out: out}
	fmt.Fprintf(w.out, "digraph %s {\n", quote(name))
	return w
}

// End writes the closing brace of the digraph.
func (w *Writer) End() {
	fmt.Fprintln(w.out, "}")
}

// Legend emits the `subgraph cluster_L` legend box used by pprof callgraphs.
func (w *Writer) Legend(title string, attrs ...Attr) {
	fmt.Fprintf(w.out, "subgraph cluster_L { %s %s }\n", quote(title), formatAttrs(attrs))
}

// Node emits a node (or default attribute, e.g. `node [...]`) statement.
func (w *Writer) Node(id ID, attrs ...Attr) {
	fmt.Fprintf(w.out, "%s %s\n", formatID(id), formatAttrs(attrs))
}

// Edge emits a directed edge statement (`from -> to [...]`).
func (w *Writer) Edge(from, to ID, attrs ...Attr) {
	fmt.Fprintf(w.out, "%s -> %s %s\n", formatID(from), formatID(to), formatAttrs(attrs))
}

func formatID(id ID) string {
	s := string(id)
	if isBareID(s) {
		return s
	}
	return quote(s)
}

func formatAttrs(attrs []Attr) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, a := range attrs {
		if i > 0 {
			b.WriteByte(' ')
		}
		if !isBareID(a.Key) {
			panic(fmt.Sprintf("invalid DOT attribute key %q", a.Key))
		}
		b.WriteString(a.Key)
		b.WriteByte('=')
		switch v := a.Val.(type) {
		case string:
			b.WriteString(quote(v))
		case Text:
			b.WriteByte('"')
			b.WriteString(v.raw)
			b.WriteByte('"')
		case ID:
			b.WriteString(formatID(v))
		case int:
			b.WriteString(strconv.Itoa(v))
		default:
			panic(fmt.Sprintf("unsupported DOT attr type %T", v))
		}
	}
	b.WriteByte(']')
	return b.String()
}

func isBareID(s string) bool {
	if s == "" || (s[0] >= '0' && s[0] <= '9') {
		return false
	}
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_':
		default:
			return false
		}
	}
	return true
}

func quote(s string) string {
	return `"` + escape(s) + `"`
}

func escapeAll(in []string) []string {
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = escape(s)
	}
	return out
}

// escape escapes double quotes and backslashes, and replaces Graphviz's
// "center" character (\n) with a left-justified character (\l).
// See https://graphviz.org/docs/attr-types/escString/ for more info.
func escape(str string) string {
	return strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(str, `\`, `\\`), `"`, `\"`), "\n", `\l`)
}
