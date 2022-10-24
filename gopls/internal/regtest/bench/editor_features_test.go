// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"testing"
)

func BenchmarkGoToDefinition(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	pos := env.RegexpSearch("internal/imports/mod.go", "ModuleJSON")
	env.GoToDefinition("internal/imports/mod.go", pos)
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.GoToDefinition("internal/imports/mod.go", pos)
	}
}

func BenchmarkFindAllReferences(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	pos := env.RegexpSearch("internal/imports/mod.go", "gopathwalk")
	env.References("internal/imports/mod.go", pos)
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.References("internal/imports/mod.go", pos)
	}
}

func BenchmarkRename(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pos := env.RegexpSearch("internal/imports/mod.go", "gopathwalk")
		newName := fmt.Sprintf("%s%d", "gopathwalk", i)
		env.Rename("internal/imports/mod.go", pos, newName)
	}
}
