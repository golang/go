// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*

9g is the version of the gc compiler for the Power64.
The $GOARCH for these tools is power64 (big endian) or
power64le (little endian).

It reads .go files and outputs .9 files. The flags are documented in ../gc/doc.go.

*/
package main
