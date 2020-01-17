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
package main
