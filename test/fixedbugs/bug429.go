// skip

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Should print deadlock message, not hang.
// This test is run by bug429_run.go.

package main

func main() {
	select {}
}
