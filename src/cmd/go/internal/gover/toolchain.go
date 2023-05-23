// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import "strings"

// ToolchainVersion returns the Go version for the named toolchain,
// derived from the name itself (not by running the toolchain).
// A toolchain is named "goVERSION" or "anything-goVERSION".
// Examples:
//
//	ToolchainVersion("go1.2.3") == "1.2.3"
//	ToolchainVersion("gccgo-go1.23rc4") == "1.23rc4"
//	ToolchainVersion("invalid") == ""
func ToolchainVersion(name string) string {
	var v string
	if strings.HasPrefix(name, "go") && IsValid(name[2:]) {
		v = name[2:]
	} else if i := strings.Index(name, "-go"); i >= 0 && IsValid(name[i+3:]) {
		v = name[i+3:]
	}
	return v
}
