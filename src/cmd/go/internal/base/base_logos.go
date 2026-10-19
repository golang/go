// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"strings"

	"cmd/go/internal/cfg"
)

// logosLongName maps a logos-mode usage line head to its long name,
// reporting false outside logos-specific names so the caller keeps
// upstream handling.
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

// helpReference returns the executable help reference naming longName,
// byte-identical to upstream text in go mode.
func helpReference(longName string) string {
	if cfg.Logos {
		return "logos help " + longName
	}
	return "go help " + longName
}
