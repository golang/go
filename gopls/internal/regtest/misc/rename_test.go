// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"strings"
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

// Test for golang/go#47564.
func TestRenameInTestVariant(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- stringutil/stringutil.go --
package stringutil

func Identity(s string) string {
	return s
}
-- stringutil/stringutil_test.go --
package stringutil

func TestIdentity(t *testing.T) {
	if got := Identity("foo"); got != "foo" {
		t.Errorf("bad")
	}
}
-- main.go --
package main

import (
	"fmt"

	"mod.com/stringutil"
)

func main() {
	fmt.Println(stringutil.Identity("hello world"))
}
`

	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		pos := env.RegexpSearch("main.go", `stringutil\.(Identity)`)
		env.Rename("main.go", pos, "Identityx")
		text := env.Editor.BufferText("stringutil/stringutil_test.go")
		if !strings.Contains(text, "Identityx") {
			t.Errorf("stringutil/stringutil_test.go: missing expected token `Identityx` after rename:\n%s", text)
		}
	})
}
