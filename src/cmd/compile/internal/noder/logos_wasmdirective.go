// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"strings"

	"cmd/compile/internal/syntax"
)

const (
	logosDirectivePrefix = "//logos:"
	goDirectivePrefix    = "//go:"
	logosWasmMarker      = logosDirectivePrefix + "wasm"
)

var logosWasmDirectives = [...]string{
	logosWasmMarker + "export",
	logosWasmMarker + "import",
}

// isLogosWasmDirective reports whether the line comment text is a
// //logos:wasmexport or //logos:wasmimport directive: the directive name
// followed by a space, a tab, or the end of the comment.
func isLogosWasmDirective(text string) bool {
	for _, name := range logosWasmDirectives {
		if rest, ok := strings.CutPrefix(text, name); ok && (rest == "" || rest[0] == ' ' || rest[0] == '\t') {
			return true
		}
	}
	return false
}

// rewriteLogosWasmDirectives spells each //logos:wasmexport and
// //logos:wasmimport line comment of a loGo/s file as its //go: form
// (SPEC Sec 14.2), so the pragma machinery, the lowered tree, and the
// ejected text all see //go:wasmexport and //go:wasmimport. It finds the
// comments with the scanner of the file's own mode, so a string literal
// that holds the same text is never rewritten. The rewrite touches only
// the directive comment, so every other byte, line, and column of the file
// stays where it was; a file with no such directive comes back unchanged.
func rewriteLogosWasmDirectives(src []byte, mode syntax.Mode) []byte {
	if !bytes.Contains(src, []byte(logosWasmMarker)) {
		return src
	}
	starts := lineStartOffsets(src)
	var hits []int
	syntax.CommentsDoMode(bytes.NewReader(src), mode, func(line, col uint, text string) {
		if isLogosWasmDirective(text) {
			hits = append(hits, starts[line-1]+int(col)-1)
		}
	})
	var out bytes.Buffer
	from := 0
	for _, at := range hits {
		out.Write(src[from:at])
		out.WriteString(goDirectivePrefix)
		from = at + len(logosDirectivePrefix)
	}
	out.Write(src[from:])
	return out.Bytes()
}

// lineStartOffsets returns the byte offset of the start of every line of
// src, the first line being line 1.
func lineStartOffsets(src []byte) []int {
	starts := []int{0}
	for i, c := range src {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}
