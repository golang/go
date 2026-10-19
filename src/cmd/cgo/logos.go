// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// isLogosCgoInput reports whether name is a loGo/s source file accepted as
// cgo input alongside .go files (M0-S5 routing: cmd/go passes logos files
// that import "C" to cmd/cgo under their .lgo name). It matches ".lgo" and
// the three alias extensions after stripping trailing U+FE0E and U+FE0F
// variation selectors, the same rule as cfg.IsLogosFile. Accepted files
// are parsed with go/parser, so only Go-syntax content compiles here;
// loGo/s syntax is rejected by the parser (mapped to
// [correctness::cgo_file_logos_syntax] by cmd/go in M20-S0). Generated
// outputs keep the input stem (for example "main.lgo.cgo1.go"), which
// satisfies the M0-S3 compiler exemption for names ending in .cgo1.go.
func isLogosCgoInput(name string) bool {
	trimmed := strings.TrimRight(name, "\uFE0E\uFE0F")
	if strings.HasSuffix(trimmed, ".lgo") {
		return true
	}
	if strings.HasSuffix(trimmed, ".📜") {
		return true
	}
	if strings.HasSuffix(trimmed, ".⚡") {
		return true
	}
	if strings.HasSuffix(trimmed, ".ʕ◔ϖ◔ʔ") {
		return true
	}
	return false
}
