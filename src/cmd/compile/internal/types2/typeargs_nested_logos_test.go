// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"fmt"
	"strings"
	"testing"
)

func TestLogosNestedTypeArgumentPositions(t *testing.T) {
	for _, typ := range []string{
		"*[]int?",
		"*([]int?)",
		"(<-chan<-chan int)?",
		"(x (a int, b string), y [2](c int, d string))",
		"(x [2][]int?, y [3]*int?)",
		"(x []map[string]int, y <-chan (int, string))",
	} {
		t.Run(typ, func(t *testing.T) {
			for _, target := range []string{"identity[argument]", "identity[" + typ + "]", "receiver{}.identity[" + typ + "]"} {
				t.Run(target, func(t *testing.T) {
					src := genericTypeArgumentPrelude + fmt.Sprintf("type argument = %s\nvar _ unary[argument] = %s\n", typ, target)
					if errs := checkGenericTypeArgumentSource(t, src); len(errs) != 0 {
						t.Fatalf("type %s via %s:\n%s", typ, target, strings.Join(errs, "\n"))
					}
				})
			}
		})
	}
}
