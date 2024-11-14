// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file holds test cases for individual issues
// for which there is (currently) no better location.

package syntax

import (
	"strings"
	"testing"
)

func TestIssue67866(t *testing.T) {
	var tests = []string{
		"package p; var _ = T{@0: 0}",
		"package p; var _ = T{@1 + 2: 0}",
		"package p; var _ = T{@x[i]: 0}",
		"package p; var _ = T{@f(1, 2, 3): 0}",
		"package p; var _ = T{@a + f(b) + <-ch: 0}",
	}

	for _, src := range tests {
		// identify column position of @ and remove it from src
		i := strings.Index(src, "@")
		if i < 0 {
			t.Errorf("%s: invalid test case (missing @)", src)
			continue
		}
		src = src[:i] + src[i+1:]
		want := colbase + uint(i)

		f, err := Parse(nil, strings.NewReader(src), nil, nil, 0)
		if err != nil {
			t.Errorf("%s: %v", src, err)
			continue
		}

		// locate KeyValueExpr
		Inspect(f, func { n ->
			_, ok := n.(*KeyValueExpr)
			if ok {
				if got := StartPos(n).Col(); got != want {
					t.Errorf("%s: got col = %d, want %d", src, got, want)
				}
			}
			return !ok
		})
	}
}
