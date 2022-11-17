// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package govulncheck

import (
	"go/token"

	"golang.org/x/vuln/osv"
)

// Vuln represents a single OSV entry.
type Vuln struct {
	// OSV contains all data from the OSV entry for this vulnerability.
	OSV *osv.Entry

	// Modules contains all of the modules in the OSV entry where a
	// vulnerable package is imported by the target source code or binary.
	//
	// For example, a module M with two packages M/p1 and M/p2, where only p1
	// is vulnerable, will appear in this list if and only if p1 is imported by
	// the target source code or binary.
	Modules []*Module
}

func (v *Vuln) IsCalled() bool {
	return false
}

// Module represents a specific vulnerability relevant to a single module.
type Module struct {
	// Path is the module path of the module containing the vulnerability.
	//
	// Importable packages in the standard library will have the path "stdlib".
	Path string

	// FoundVersion is the module version where the vulnerability was found.
	FoundVersion string

	// FixedVersion is the module version where the vulnerability was
	// fixed. If there are multiple fixed versions in the OSV report, this will
	// be the latest fixed version.
	//
	// This is empty if a fix is not available.
	FixedVersion string

	// Packages contains all the vulnerable packages in OSV entry that are
	// imported by the target source code or binary.
	//
	// For example, given a module M with two packages M/p1 and M/p2, where
	// both p1 and p2 are vulnerable, p1 and p2 will each only appear in this
	// list they are individually imported by the target source code or binary.
	Packages []*Package
}

// Package is a Go package with known vulnerable symbols.
type Package struct {
	// Path is the import path of the package containing the vulnerability.
	Path string

	// CallStacks contains a representative call stack for each
	// vulnerable symbol that is called.
	//
	// For vulnerabilities found from binary analysis, only CallStack.Symbol
	// will be provided.
	//
	// For non-affecting vulnerabilities reported from the source mode
	// analysis, this will be empty.
	CallStacks []CallStack
}

// CallStacks contains a representative call stack for a vulnerable
// symbol.
type CallStack struct {
	// Symbol is the name of the detected vulnerable function
	// or method.
	//
	// This follows the naming convention in the OSV report.
	Symbol string

	// Summary is a one-line description of the callstack, used by the
	// default govulncheck mode.
	//
	// Example: module3.main calls github.com/shiyanhui/dht.DHT.Run
	Summary string

	// Frames contains an entry for each stack in the call stack.
	//
	// Frames are sorted starting from the entry point to the
	// imported vulnerable symbol. The last frame in Frames should match
	// Symbol.
	Frames []*StackFrame
}

// StackFrame represents a call stack entry.
type StackFrame struct {
	// PackagePath is the import path.
	PkgPath string

	// FuncName is the function name.
	FuncName string

	// RecvType is the fully qualified receiver type,
	// if the called symbol is a method.
	//
	// The client can create the final symbol name by
	// prepending RecvType to FuncName.
	RecvType string

	// Position describes an arbitrary source position
	// including the file, line, and column location.
	// A Position is valid if the line number is > 0.
	Position token.Position
}

func (sf *StackFrame) Name() string {
	return ""
}

func (sf *StackFrame) Pos() string {
	return ""
}
