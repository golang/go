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
// The parser accepts a minimal TOML subset: the [project], [build],
// [capabilities], [lints], and [lints.thresholds] tables, bare keys, basic
// and literal strings, decimal integers (thresholds only), string arrays
// (including multi-line arrays with trailing commas), and hash comments. It
// rejects everything outside the schema. Every error reports file, line, and
// column as name:line:col: message, where name is the name argument passed
// to Parse.
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
	// Capabilities holds the [capabilities] table.
	Capabilities Capabilities
	// Lints holds the [lints] and [lints.thresholds] tables.
	Lints Lints
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

// Build holds the [build] table: the cgo policy and its allowlist, the
// optimization level, the synthetic PGO level, the CPU target, the
// arena experiment, and the collector mode.
type Build struct {
	// Cgo is one of "forbid", "restricted", or "allow". It defaults to "forbid".
	Cgo string
	// CgoSet reports whether the cgo key was present in the file.
	CgoSet bool
	// CgoAllowlist holds path globs enabled under the "restricted" policy.
	CgoAllowlist []string
	// OptLevel is one of "O0", "O1", "O2", "O3", "Os", or "Oz". It defaults to "O2".
	OptLevel string
	// SyntheticPGO is one of "off", "standard", or "aggressive".
	SyntheticPGO string
	// SynthPGOSet reports whether the synthetic_pgo key was present in the file.
	SynthPGOSet bool
	// TargetCPU is one of "deterministic", "baseline", or "native". It defaults to "deterministic".
	TargetCPU string
	// Arena enables the tier-0 arena experiment (GOEXPERIMENT=logosarena)
	// for the build. It defaults to false.
	Arena bool
	// GC is the collector mode named by the gc key. It is checked against
	// GCModes at build time, with the same text as the -gc flag, and is empty
	// while GCSet is false.
	GC string
	// GCSet reports whether the gc key was present in the file.
	GCSet bool
}

// GCModes lists the collector modes that -gc and the gc key accept, in the
// order the unknown-mode message names them.
var GCModes = []string{"greentea", "zgc", "immix", "wasm", "batch", "epsilon"}

// Capabilities holds the [capabilities] table (M12): severity keys, each one
// of "deny", "warn", or "allow", and allowlist keys holding path globs.
// Severities default to deny, except TransitiveReflect which defaults to
// warn; allowlists default to empty.
type Capabilities struct {
	// DirectReflect guards direct reachability of reflect sinks.
	DirectReflect string
	// TransitiveReflect guards transitive reachability of reflect sinks.
	TransitiveReflect string
	// TransitiveExec guards transitive reachability of exec sinks.
	TransitiveExec string
	// TransitiveNet guards transitive reachability of net sinks.
	TransitiveNet string
	// TransitiveUnsafe guards transitive imports of "unsafe".
	TransitiveUnsafe string
	// ReflectAllowlist holds path globs exempt from reflect checks.
	ReflectAllowlist []string
	// TransitiveExecAllowlist holds path globs exempt from exec checks.
	TransitiveExecAllowlist []string
	// TransitiveNetAllowlist holds path globs exempt from net checks.
	TransitiveNetAllowlist []string
	// TransitiveUnsafeAllowlist holds path globs exempt from unsafe checks.
	TransitiveUnsafeAllowlist []string
}

// strictFileExts lists every loGo/s source extension: .lgo, the scroll
// (U+1F4DC), the lightning bolt (U+26A1), and the gopher extension
// (U+0295 U+25D4 U+03D6 U+25D4 U+0294).
var strictFileExts = []string{".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

// defaultFile returns the File used when logos.toml (or a table or key) is
// absent: edition "2026", all four extensions enabled, cgo "forbid" with the
// key marked absent, opt_level "O2", no synthetic_pgo key, and target_cpu
// "deterministic".
func defaultFile() *File {
	return &File{
		Project: Project{Edition: "2026", StrictFiles: slices.Clone(strictFileExts)},
		Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		Capabilities: Capabilities{
			DirectReflect:     "deny",
			TransitiveReflect: "warn",
			TransitiveExec:    "deny",
			TransitiveNet:     "deny",
			TransitiveUnsafe:  "deny",
		},
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
// re-parse), an unset cgo or gc key (which must keep CgoSet or GCSet false),
// and nil allowlists.
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
	if f.Build.OptLevel != "" && f.Build.OptLevel != "O2" {
		b.WriteString("opt_level = ")
		appendQuoted(&b, f.Build.OptLevel)
		b.WriteByte('\n')
	}
	if f.Build.SynthPGOSet {
		b.WriteString("synthetic_pgo = ")
		appendQuoted(&b, f.Build.SyntheticPGO)
		b.WriteByte('\n')
	}
	if f.Build.TargetCPU != "" && f.Build.TargetCPU != "deterministic" {
		b.WriteString("target_cpu = ")
		appendQuoted(&b, f.Build.TargetCPU)
		b.WriteByte('\n')
	}
	if f.Build.Arena {
		b.WriteString("arena = true\n")
	}
	if f.Build.GCSet {
		b.WriteString("gc = ")
		appendQuoted(&b, f.Build.GC)
		b.WriteByte('\n')
	}
	writeCapabilities(&b, f.Capabilities)
	return []byte(b.String())
}

// writeCapabilities renders the [capabilities] table, omitting severities
// at their defaults and nil allowlists so Format output re-parses to an
// equal File.
func writeCapabilities(b *strings.Builder, c Capabilities) {
	sev := func(key, v, def string) {
		if v != "" && v != def {
			b.WriteString(key)
			b.WriteString(" = ")
			appendQuoted(b, v)
			b.WriteByte('\n')
		}
	}
	list := func(key string, vs []string) {
		if vs == nil {
			return
		}
		b.WriteString(key)
		b.WriteString(" = [")
		for i, s := range vs {
			if i > 0 {
				b.WriteString(", ")
			}
			appendQuoted(b, s)
		}
		b.WriteString("]\n")
	}
	emit := c.DirectReflect != "" && c.DirectReflect != "deny" ||
		c.TransitiveReflect != "" && c.TransitiveReflect != "warn" ||
		c.TransitiveExec != "" && c.TransitiveExec != "deny" ||
		c.TransitiveNet != "" && c.TransitiveNet != "deny" ||
		c.TransitiveUnsafe != "" && c.TransitiveUnsafe != "deny" ||
		c.ReflectAllowlist != nil || c.TransitiveExecAllowlist != nil ||
		c.TransitiveNetAllowlist != nil || c.TransitiveUnsafeAllowlist != nil
	if !emit {
		return
	}
	b.WriteString("[capabilities]\n")
	sev("direct_reflect", c.DirectReflect, "deny")
	sev("transitive_reflect", c.TransitiveReflect, "warn")
	sev("transitive_exec", c.TransitiveExec, "deny")
	sev("transitive_net", c.TransitiveNet, "deny")
	sev("transitive_unsafe", c.TransitiveUnsafe, "deny")
	list("reflect_allowlist", c.ReflectAllowlist)
	list("transitive_exec_allowlist", c.TransitiveExecAllowlist)
	list("transitive_net_allowlist", c.TransitiveNetAllowlist)
	list("transitive_unsafe_allowlist", c.TransitiveUnsafeAllowlist)
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
