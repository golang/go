// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"testing"
)

func BenchmarkRename(b *testing.B) {
	env := repos["tools"].sharedEnv(b)

	env.OpenFile("internal/imports/mod.go")
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 1; i < b.N; i++ {
		loc := env.RegexpSearch("internal/imports/mod.go", "gopathwalk")
		newName := fmt.Sprintf("%s%d", "gopathwalk", i)
		env.Rename(loc, newName)
	}
}
