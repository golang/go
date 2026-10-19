// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"os"
	"path/filepath"
	"strings"
)

// Logos reports whether the current process was invoked as the logos
// command. The logos command is the same binary as the go command,
// selected by argv[0]: the base name of os.Args[0] with a ".exe"
// suffix removed must equal "logos". It is a package-level initializer,
// so defaultContext observes it when BuildContext is first built.
var Logos = strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") == "logos"

// Pedantic opts in to pedantic diagnostics for logos builds. It is set
// by the logos-only "logos build --pedantic" flag (registered in
// cmd/go/internal/logos.ConfigureCommand) and overrides the logos.toml
// pedantic policy (M9); this slice only stores it. Plain go never sets
// it: the flag is undefined there, so "go help build" keeps the
// upstream text.
var Pedantic bool

// logosToolTags returns tags plus the "logos" tool tag in logos mode.
// defaultContext calls it, so every rebuilt BuildContext keeps the tag.
func logosToolTags(tags []string) []string {
	if !Logos {
		return tags
	}
	return append(tags, "logos")
}

// IsLogosFile reports whether name is a loGo/s source file. It is
// mode-independent: it matches ".lgo" and the three alias extensions
// after stripping trailing U+FE0E and U+FE0F variation selectors.
// It reads no mode state, so it may be called at any time.
func IsLogosFile(name string) bool {
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

// IsGoFile reports whether name is a Go-family source file: a ".go"
// file, or a loGo/s file when the current process is the logos command.
// It is valid only at run time, after init has set Logos.
func IsGoFile(name string) bool {
	if strings.HasSuffix(name, ".go") {
		return true
	}
	return Logos && IsLogosFile(name)
}
