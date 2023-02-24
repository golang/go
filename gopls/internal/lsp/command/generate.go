// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"log"
	"os"

	"golang.org/x/tools/gopls/internal/lsp/command/gen"
)

func main() {
	content, err := gen.Generate()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("command_gen.go", content, 0644); err != nil {
		log.Fatal(err)
	}
}
