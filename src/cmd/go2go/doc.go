// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go2go is a command for trying out generic Go code.
// It supports a small number of commands similar to cmd/go.
//
// Usage:
//
//	go2go <command> [arguments]
//
// The commands are:
//
//	build      translate and then run "go build packages"
//	run        translate and then run a list of files
//      test       translate and then run "go test packages"
//      translate  translate .go2 files into .go files for listed packages
//
// A package is expected to contain .go2 files but no .go files.
//
// Non-local imported packages will be first looked up using the GO2PATH
// environment variable, which should point to a GOPATH-like directory.
// For example, import "x" will first look for $GO2PATH/src/x.
// If not found in GO2PATH, imports will be looked up in the usual way.
// If an import includes .go2 files, it will be translated into .go files.
//
// Translation into standard Go requires generating Go code with mangled names.
// The mangled names will always include Odia (Oriya) digits, such as ୦ and ୮.
// Do not use Oriya digits in identifiers in your own code.
package main
