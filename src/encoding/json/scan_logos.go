// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"internal/jsonscan"
	"unicode/utf8"
)

const hotSkip = 32

func skipEscapeHTML[Bytes []byte | string](b Bytes, n int) (next int, hot bool) {
	k := jsonscan.IndexEscapeHTML(b[n:])
	return n + k, k >= hotSkip
}

func skipStringSpecial(b []byte, n int) (next int, hot bool) {
	k := jsonscan.IndexStringSpecial(b[n:])
	return n + k, k >= hotSkip
}

func skipPlainAfter(b []byte, i int) (last int, hot bool) {
	if i >= len(b) {
		return i, false
	}
	k := jsonscan.IndexStringSpecial(b[i+1:])
	return i + k, k >= hotSkip
}

func plainRun(data []byte, hot bool) (n int, next bool) {
	if hot {
		n, hot = skipStringSpecial(data, n)
	}
	for n < len(data) {
		c := data[n]
		if c == '"' || c == '\\' || c < ' ' {
			return n, hot
		}
		n++
		if c >= utf8.RuneSelf && hot {
			n, hot = skipStringSpecial(data, n)
		}
	}
	return n, hot
}

func (s *scanner) stepEscape(data []byte) int {
	n := 2
	if len(data) > 1 && data[1] == 'u' {
		n = 6
	}
	n = min(n, len(data))
	for i := range n {
		s.bytes++
		if s.step(s, data[i]) == scanError {
			return i + 1
		}
	}
	return n
}

func (s *scanner) skipStringBody(data []byte) int {
	i, hot := 0, true
	for i < len(data) {
		n, next := plainRun(data[i:], hot)
		i += n
		hot = next
		s.bytes += int64(n)
		if i == len(data) || data[i] != '\\' {
			break
		}
		i += s.stepEscape(data[i:])
		if s.err != nil {
			break
		}
	}
	return i
}
