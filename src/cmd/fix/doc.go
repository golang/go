// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Fix finds Go programs that use old APIs and rewrites them to use
newer ones.  After you update to a new Go release, fix helps make
the necessary changes to your programs.

Usage:

	go tool fix [ignored...]

This tool is currently in transition. All its historical fixers were
long obsolete and have been removed, so it is currently a no-op. In
due course the tool will integrate with the Go analysis framework
(golang.org/x/tools/go/analysis) and run a modern suite of fix
algorithms; see https://go.dev/issue/71859.
*/
package main
