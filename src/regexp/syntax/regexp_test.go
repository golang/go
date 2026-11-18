// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "testing"

var benchmarkTests = []string{
	`^(.*);$|^;(.*)`,
	`(foo|bar$)x*`,
	`[^=,]`,
	`([^=,]+)=([^=,]+)`,
	`([^=,]+)=([^=,]+),.*`,
}

func BenchmarkString(b *testing.B) {
	for _, tt := range benchmarkTests {
		re, err := Parse(tt, Perl|DotNL)
		if err != nil {
			b.Fatalf("Parse(%#q) = error %v", tt, err)
		}

		b.Run(tt, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = re.String()
			}
		})
	}
}
