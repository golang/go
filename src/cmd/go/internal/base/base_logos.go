// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"internal/logosfile"
	"strings"

	"cmd/go/internal/cfg"
)

func logosLongName(name string) (string, bool) {
	if !cfg.Logos {
		return "", false
	}
	if name == "logos" {
		return "", true
	}
	if rest, ok := strings.CutPrefix(name, "logos "); ok {
		return rest, true
	}
	return "", false
}

func helpReference(longName string) string {
	if cfg.Logos {
		return "logos help " + longName
	}
	return "go help " + longName
}

func isLogosTestFile(file string) bool {
	if !logosfile.Is(file) {
		return false
	}
	stem := file
	if cut, _, found := strings.CutLast(stem, "."); found {
		stem = cut
	}
	return strings.HasSuffix(stem, "_test")
}
