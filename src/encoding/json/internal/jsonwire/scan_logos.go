// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"encoding/json/internal/jsonflags"
	"internal/jsonscan"
)

const hotSkip = 32

func skipEscapeHTML(b []byte, n int) (next int, hot bool) {
	k := jsonscan.IndexEscapeHTML(b[n:])
	return n + k, k >= hotSkip
}

func skipEscape(b []byte, n int, flags *jsonflags.Flags) (next int, hot bool) {
	if flags.Get(jsonflags.EscapeForHTML) {
		return skipEscapeHTML(b, n)
	}
	k := jsonscan.IndexEscape(b[n:])
	return n + k, k >= hotSkip
}

func skipStringSpecial(b []byte, n int) (next int, hot bool) {
	k := jsonscan.IndexStringSpecial(b[n:])
	return n + k, k >= hotSkip
}
