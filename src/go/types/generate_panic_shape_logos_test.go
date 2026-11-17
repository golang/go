// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGoTypesPanicSourceShape(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(runtime.GOROOT(), "src/cmd/compile/internal/types2/builtins.go"))
	if err != nil {
		t.Fatal(err)
	}
	for count := 0; count <= 2; count++ {
		t.Run(fmt.Sprintf("occurrences=%d", count), func(t *testing.T) {
			defer func() {
				failure := recover()
				if count == 1 {
					if failure != nil {
						t.Fatalf("single compiler panic guard rejected: %v", failure)
					}
					return
				}
				message, ok := failure.(string)
				if !ok || !strings.Contains(message, "expected exactly one loGo/s panic registration guard") {
					t.Fatalf("guard count %d: got panic %v, want descriptive shape rejection", count, failure)
				}
			}()
			goTypesPanicSource(bytes.Repeat(src, count))
		})
	}
}
