// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package main_test

import "testing"

func TestBoringInternalLink(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("main.go", `package main
		import "crypto/sha1"
		func main() {
			sha1.New()
		}`)
	tg.run("build", "-ldflags=-w -extld=false", tg.path("main.go"))
	tg.run("build", "-ldflags=-extld=false", tg.path("main.go"))
}
