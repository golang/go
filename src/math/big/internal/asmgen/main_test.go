// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

import (
	"bytes"
	"flag"
	"internal/diff"
	"os"
	"testing"
)

var generateFlag = flag.Bool("generate", false, "generate files")

func Test(t *testing.T) {
	for _, arch := range arches {
		t.Run(arch.Name, func(t *testing.T) {
			file, data := generate(arch)
			old, err := os.ReadFile("../../" + file)
			if err == nil && bytes.Equal(old, data) {
				return
			}
			if *generateFlag {
				if err := os.WriteFile("../../"+file, data, 0o666); err != nil {
					t.Fatal(err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Fatalf("generated assembly differs:\n%s\n", diff.Diff("../../"+file, old, "regenerated", data))
		})
	}
}
