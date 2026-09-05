// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constraint_test

import (
	"fmt"
	"log"

	"go/build/constraint"
)

func ExampleParse() {
	expr, err := constraint.Parse("//go:build linux && (amd64 || arm64)")
	if err != nil {
		log.Fatal(err)
	}

	tagSets := []map[string]bool{
		{"linux": true, "arm64": true},
		{"linux": true, "amd64": true},
		{"linux": true},
		{"arm64": true},
	}

	fmt.Println(expr)
	for _, tags := range tagSets {
		fmt.Println(expr.Eval(func(tag string) bool {
			return tags[tag]
		}))
	}

	// Output:
	// linux && (amd64 || arm64)
	// true
	// true
	// false
	// false
}
