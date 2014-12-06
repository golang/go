// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*

9g is the version of the gc compiler for 64-bit PowerPC or Power Architecture processors.
The $GOARCH for these tools is ppc64 (big endian) or
ppc64le (little endian).

It reads .go files and outputs .9 files. The flags are documented in ../gc/doc.go.

*/
package main
