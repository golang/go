// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// # ! $G $D/$F.go

package main

foo

func main() {}

/*
bug25.go:5: fatal error: dumpexportvar: oname nil: Foo

*/
