// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// autovec_overlay constructs one exact mutation without editing the checkout.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	if len(os.Args) != 5 || os.Args[2] == "" {
		panic("usage: autovec_overlay source old new output-directory")
	}
	source, err := filepath.Abs(os.Args[1])
	if err != nil {
		panic(err)
	}
	data, err := os.ReadFile(source)
	if err != nil {
		panic(err)
	}
	old, replacement := os.Args[2], os.Args[3]
	if count := strings.Count(string(data), old); count != 1 {
		panic(fmt.Sprintf("mutation must match exactly once, got %d", count))
	}
	target, err := filepath.Abs(filepath.Join(os.Args[4], filepath.Base(source)))
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(target, []byte(strings.Replace(string(data), old, replacement, 1)), 0600); err != nil {
		panic(err)
	}
	overlay := struct {
		Replace map[string]string
	}{Replace: map[string]string{source: target}}
	if err := json.NewEncoder(os.Stdout).Encode(overlay); err != nil {
		panic(err)
	}
}
