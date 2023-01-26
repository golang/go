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
	loc := env.RegexpSearch("internal/imports/mod.go", "ModuleJSON")
	env.GoToDefinition(loc)
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.GoToDefinition(loc)
	}
}

func BenchmarkFindAllReferences(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	loc := env.RegexpSearch("internal/imports/mod.go", "gopathwalk")
	env.References(loc)
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.References(loc)
	}
}

func BenchmarkRename(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 1; i < b.N; i++ {
		loc := env.RegexpSearch("internal/imports/mod.go", "gopathwalk")
		newName := fmt.Sprintf("%s%d", "gopathwalk", i)
		env.Rename(loc, newName)
	}
}

func BenchmarkFindAllImplementations(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	loc := env.RegexpSearch("internal/imports/mod.go", "initAllMods")
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.Implementations(loc)
	}
}

func BenchmarkHover(b *testing.B) {
	env := benchmarkEnv(b)

	env.OpenFile("internal/imports/mod.go")
	loc := env.RegexpSearch("internal/imports/mod.go", "bytes")
	env.Await(env.DoneWithOpen())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.Hover(loc)
	}
}
