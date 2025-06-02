// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token_test

import (
	"go/build"
	"go/token"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkSearchInts(b *testing.B) {
	data := make([]int, 10000)
	for i := 0; i < 10000; i++ {
		data[i] = i
	}
	const x = 8
	if r := token.SearchInts(data, x); r != x {
		b.Errorf("got index = %d; want %d", r, x)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		token.SearchInts(data, x)
	}
}

func BenchmarkFileSet_Position(b *testing.B) {
	rng := rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))

	// Create a FileSet based on the files of net/http,
	// a single large package.
	netHTTPFset := token.NewFileSet()
	pkg, err := build.Default.Import("net/http", "", 0)
	if err != nil {
		b.Fatal(err)
	}
	for _, filename := range pkg.GoFiles {
		filename = filepath.Join(pkg.Dir, filename)
		fi, err := os.Stat(filename)
		if err != nil {
			b.Fatal(err)
		}
		netHTTPFset.AddFile(filename, -1, int(fi.Size()))
	}

	// Measure randomly distributed Pos values across net/http.
	b.Run("random", func(b *testing.B) {
		base := netHTTPFset.Base()
		for b.Loop() {
			pos := token.Pos(rng.IntN(base))
			_ = netHTTPFset.Position(pos)
		}
	})

	// Measure random lookups within the same file of net/http.
	// (It's faster because of the "last file" cache.)
	b.Run("file", func(b *testing.B) {
		var file *token.File
		for file = range netHTTPFset.Iterate {
			break
		}
		base, size := file.Base(), file.Size()
		for b.Loop() {
			_ = netHTTPFset.Position(token.Pos(base + rng.IntN(size)))
		}
	})

	// Measure random lookups on a FileSet with a great many files.
	b.Run("manyfiles", func(b *testing.B) {
		fset := token.NewFileSet()
		for range 25000 {
			fset.AddFile("", -1, 10000)
		}
		base := fset.Base()
		for b.Loop() {
			pos := token.Pos(rng.IntN(base))
			_ = fset.Position(pos)
		}
	})
}

func BenchmarkFileSet_AddExistingFiles(b *testing.B) {
	// Create the "universe" of files.
	fset := token.NewFileSet()
	var files []*token.File
	for range 25000 {
		files = append(files, fset.AddFile("", -1, 10000))
	}
	rand.Shuffle(len(files), func(i, j int) {
		files[i], files[j] = files[j], files[i]
	})

	// choose returns n random files.
	choose := func(n int) []*token.File {
		res := make([]*token.File, n)
		for i := range res {
			res[i] = files[rand.IntN(n)]
		}
		return files[:n]
	}

	// Measure the cost of	creating a FileSet with a large number
	// of files added in small handfuls, with some overlap.
	// This case is critical to gopls.
	b.Run("sequence", func(b *testing.B) {
		for b.Loop() {
			b.StopTimer()
			fset2 := token.NewFileSet()
			fset2.AddExistingFiles(files[:10000]...)
			b.StartTimer()

			for range 1000 {
				fset2.AddExistingFiles(choose(10)...) // about one package
			}
		}
	})
}
