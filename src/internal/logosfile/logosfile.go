// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package logosfile is the only place that lists the loGo/s source
// extensions. Trailing U+FE0E and U+FE0F variation selectors do not
// count: "x.⚡" and "x.⚡\uFE0F" are the same kind of file.
package logosfile

import (
	"slices"
	"strings"
)

// Canonical is the extension that loGo/s tools write.
const Canonical = ".lgo"

var exts = [...]string{Canonical, ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

const variationSelectors = "\uFE0E\uFE0F"

// Exts returns every loGo/s source extension, dot included, canonical first.
func Exts() []string {
	return slices.Clone(exts[:])
}

// Ext returns the loGo/s extension of name as Exts lists it, without
// variation selectors. It returns "" when name is not loGo/s source.
func Ext(name string) string {
	trimmed := strings.TrimRight(name, variationSelectors)
	for _, e := range exts {
		if strings.HasSuffix(trimmed, e) {
			return e
		}
	}
	return ""
}

// Split cuts name at its loGo/s extension. The returned extension is the
// on-disk one, variation selectors included, so stem+ext == name.
func Split(name string) (stem, ext string, ok bool) {
	e := Ext(name)
	if e == "" {
		return "", "", false
	}
	stem = name[:len(strings.TrimRight(name, variationSelectors))-len(e)]
	return stem, name[len(stem):], true
}

// Is reports whether name, a base name, a path or a bare extension, is
// loGo/s source.
func Is(name string) bool {
	return Ext(name) != ""
}

// IsAlias reports whether name is loGo/s source in one of the alias
// extensions, not the canonical one.
func IsAlias(name string) bool {
	e := Ext(name)
	return e != "" && e != Canonical
}
