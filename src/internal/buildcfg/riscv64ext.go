// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// To add a new extension:
// 1. Add a constant for the extension name (e.g., Riscv64ExtNewExt = "newext")
// 2. Add a field to Goriscv64Extensions struct (e.g., NewExt bool)
// 3. Add an entry to this registry with the name and a function that returns &g.NewExt

package buildcfg

import "strings"

// RISC-V 64-bit extension name constants.
const (
	Riscv64ExtZacas = "zacas"
	Riscv64ExtZabha = "zabha"
)

// riscv64ExtInfo contains information about a RISC-V 64-bit extension.
type riscv64ExtInfo struct {
	name     string
	accessor func(*Goriscv64Extensions) *bool
}

// riscv64ExtRegistry is the registry of all supported RISC-V 64-bit extensions.
var riscv64ExtRegistry = []riscv64ExtInfo{
	{Riscv64ExtZacas, func(g *Goriscv64Extensions) *bool { return &g.Zacas }},
	{Riscv64ExtZabha, func(g *Goriscv64Extensions) *bool { return &g.Zabha }},
}

// Goriscv64Extensions represents the enabled RISC-V 64-bit extensions.
type Goriscv64Extensions struct {
	Zacas bool
	Zabha bool
}

// Has returns true if the given extension is enabled.
func (g Goriscv64Extensions) Has(ext string) bool {
	for _, info := range riscv64ExtRegistry {
		if info.name == ext {
			return *info.accessor(&g)
		}
	}
	return false
}

// String returns a comma-separated list of enabled extensions.
func (g Goriscv64Extensions) String() string {
	var flags []string
	for _, info := range riscv64ExtRegistry {
		if *info.accessor(&g) {
			flags = append(flags, info.name)
		}
	}
	return strings.Join(flags, ",")
}

// EnabledExtensions returns a slice of enabled extension names.
func (g Goriscv64Extensions) EnabledExtensions() []string {
	var exts []string
	for _, info := range riscv64ExtRegistry {
		if *info.accessor(&g) {
			exts = append(exts, info.name)
		}
	}
	return exts
}

// isValidRiscv64Ext checks if the given extension name is a valid RISC-V 64-bit extension.
func isValidRiscv64Ext(ext string) bool {
	for _, info := range riscv64ExtRegistry {
		if info.name == ext {
			return true
		}
	}
	return false
}

// allowedRiscv64OptList returns a comma-separated list of all supported extension names.
func allowedRiscv64OptList() string {
	names := make([]string, len(riscv64ExtRegistry))
	for i, info := range riscv64ExtRegistry {
		names[i] = info.name
	}
	return strings.Join(names, ", ")
}

// goriscv64Extensions extracts extensions from GORISCV64 environment variable.
// Format: GORISCV64="rva23u64,zacas,zabha" -> returns struct with Zacas and Zabha set to true.
// returns zero struct if there is an error
func goriscv64Extensions() Goriscv64Extensions {
	v := envOr("GORISCV64", DefaultGORISCV64)
	_, extensions, err := ParseGORISCV64(v)
	if err != nil {
		Error = err
		return Goriscv64Extensions{}
	}

	var ext Goriscv64Extensions
	for extName, enabled := range extensions {
		for _, info := range riscv64ExtRegistry {
			if info.name == extName {
				*info.accessor(&ext) = enabled
				break
			}
		}
	}
	return ext
}
