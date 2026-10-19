// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package config reads logos.toml, the loGo/s module configuration file.
//
// logos.toml sits next to go.mod in the main module. In logos mode the go
// command loads it for the main module containing the current directory;
// plain go reads no logos.toml. When the file is absent every key takes its
// default.
//
// The parser accepts a minimal TOML subset: the [project] and [build]
// tables, bare keys, basic and literal strings, string arrays (including
// multi-line arrays with trailing commas), and hash comments. It rejects
// everything outside the schema. Every error reports file, line, and column
// as name:line:col: message, where name is the name argument passed to
// Parse.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// File is a parsed logos.toml document.
type File struct {
	// Project holds the [project] table.
	Project Project
	// Build holds the [build] table.
	Build Build
}

// Project holds the [project] table: the module name (informational only),
// the language edition, and the enabled loGo/s source extensions.
type Project struct {
	// Name is informational and changes no build output. It must not be empty.
	Name string
	// Edition selects the language rules. Only "2026" is valid.
	Edition string
	// StrictFiles lists the enabled loGo/s extensions. It defaults to all four.
	StrictFiles []string
}

// Build holds the [build] table: the cgo policy and its allowlist.
type Build struct {
	// Cgo is one of "forbid", "restricted", or "allow". It defaults to "forbid".
	Cgo string
	// CgoSet reports whether the cgo key was present in the file.
	CgoSet bool
	// CgoAllowlist holds path globs enabled under the "restricted" policy.
	CgoAllowlist []string
}

// strictFileExts lists every loGo/s source extension: .lgo, the scroll
// (U+1F4DC), the lightning bolt (U+26A1), and the gopher extension
// (U+0295 U+25D4 U+03D6 U+25D4 U+0294).
var strictFileExts = []string{".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

// defaultFile returns the File used when logos.toml (or a table or key) is
// absent: edition "2026", all four extensions enabled, cgo "forbid" with the
// key marked absent.
func defaultFile() *File {
	return &File{
		Project: Project{Edition: "2026", StrictFiles: slices.Clone(strictFileExts)},
		Build:   Build{Cgo: "forbid"},
	}
}

// Load reads logos.toml from modRoot. An empty modRoot or an absent file
// yields defaults. A present file is parsed with Parse. Read failures
// (other than absence) are prefixed with the logos.toml path.
func Load(modRoot string) (*File, error) {
	if modRoot == "" {
		return defaultFile(), nil
	}
	path := filepath.Join(modRoot, "logos.toml")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultFile(), nil
		}
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return Parse("logos.toml", data)
}

// Format renders f as canonical TOML that re-parses to an equal File.
// Keys with default-absent state are omitted: an empty name (which could not
// re-parse), an unset cgo key (which must keep CgoSet false), and nil
// allowlists.
func (f *File) Format() []byte {
	var b strings.Builder
	b.WriteString("[project]\n")
	if f.Project.Name != "" {
		b.WriteString("name = ")
		appendQuoted(&b, f.Project.Name)
		b.WriteByte('\n')
	}
	if f.Project.Edition != "" {
		b.WriteString("edition = ")
		appendQuoted(&b, f.Project.Edition)
		b.WriteByte('\n')
	}
	if f.Project.StrictFiles != nil {
		b.WriteString("strict_files = [")
		for i, s := range f.Project.StrictFiles {
			if i > 0 {
				b.WriteString(", ")
			}
			appendQuoted(&b, s)
		}
		b.WriteString("]\n")
	}
	b.WriteString("[build]\n")
	if f.Build.CgoSet {
		b.WriteString("cgo = ")
		appendQuoted(&b, f.Build.Cgo)
		b.WriteByte('\n')
	}
	if f.Build.CgoAllowlist != nil {
		b.WriteString("cgo_allowlist = [")
		for i, s := range f.Build.CgoAllowlist {
			if i > 0 {
				b.WriteString(", ")
			}
			appendQuoted(&b, s)
		}
		b.WriteString("]\n")
	}
	return []byte(b.String())
}

// appendQuoted writes s as a TOML basic string, escaping quotes, backslashes,
// and control characters (including DEL) and leaving all other runes
// (including multibyte extensions) raw.
func appendQuoted(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString("\\\"")
		case '\\':
			b.WriteString("\\\\")
		case '\b':
			b.WriteString("\\b")
		case '\t':
			b.WriteString("\\t")
		case '\n':
			b.WriteString("\\n")
		case '\f':
			b.WriteString("\\f")
		case '\r':
			b.WriteString("\\r")
		default:
			if r < 0x20 || r == 0x7F {
				const hex = "0123456789ABCDEF"
				b.WriteString("\\u00")
				b.WriteByte(hex[byte(r)>>4])
				b.WriteByte(hex[byte(r)&0x0f])
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}
