// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"

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

func isLogosWasmDirective(text string) bool {
	for _, name := range logosWasmDirectives {
		if _, ok := syntax.DirectiveArg(text, name); ok {
			return true
		}
	}
	return false
}

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

func lineStartOffsets(src []byte) []int {
	starts := []int{0}
	for i, c := range src {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}
