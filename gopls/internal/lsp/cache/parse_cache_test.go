// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/token"
	"math/bits"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

func skipIfNoParseCache(t *testing.T) {
	if bits.UintSize == 32 {
		t.Skip("the parse cache is not supported on 32-bit systems")
	}
}

func TestParseCache(t *testing.T) {
	skipIfNoParseCache(t)

	ctx := context.Background()
	uri := span.URI("file:///myfile")
	fh := makeFakeFileHandle(uri, []byte("package p\n\nconst _ = \"foo\""))
	fset := token.NewFileSet()

	cache := newParseCache(0)
	pgfs1, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh)
	if err != nil {
		t.Fatal(err)
	}
	pgf1 := pgfs1[0]
	pgfs2, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh)
	pgf2 := pgfs2[0]
	if err != nil {
		t.Fatal(err)
	}
	if pgf1 != pgf2 {
		t.Errorf("parseFiles(%q): unexpected cache miss on repeated call", uri)
	}

	// Fill up the cache with other files, but don't evict the file above.
	cache.gcOnce()
	files := []source.FileHandle{fh}
	files = append(files, dummyFileHandles(parseCacheMinFiles-1)...)

	pgfs3, err := cache.parseFiles(ctx, fset, source.ParseFull, false, files...)
	if err != nil {
		t.Fatal(err)
	}
	pgf3 := pgfs3[0]
	if pgf3 != pgf1 {
		t.Errorf("parseFiles(%q, ...): unexpected cache miss", uri)
	}
	if pgf3.Tok.Base() != pgf1.Tok.Base() || pgf3.Tok.Size() != pgf1.Tok.Size() {
		t.Errorf("parseFiles(%q, ...): result.Tok has base: %d, size: %d, want (%d, %d)", uri, pgf3.Tok.Base(), pgf3.Tok.Size(), pgf1.Tok.Base(), pgf1.Tok.Size())
	}
	if tok := fset.File(token.Pos(pgf3.Tok.Base())); tok != pgf3.Tok {
		t.Errorf("parseFiles(%q, ...): result.Tok not contained in FileSet", uri)
	}

	// Now overwrite the cache, after which we should get new results.
	cache.gcOnce()
	files = dummyFileHandles(parseCacheMinFiles)
	_, err = cache.parseFiles(ctx, fset, source.ParseFull, false, files...)
	if err != nil {
		t.Fatal(err)
	}
	// force a GC, which should collect the recently parsed files
	cache.gcOnce()
	pgfs4, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh)
	if err != nil {
		t.Fatal(err)
	}
	if pgfs4[0] == pgf1 {
		t.Errorf("parseFiles(%q): unexpected cache hit after overwriting cache", uri)
	}
}

func TestParseCache_Reparsing(t *testing.T) {
	skipIfNoParseCache(t)

	defer func(padding int) {
		parsePadding = padding
	}(parsePadding)
	parsePadding = 0

	files := dummyFileHandles(parseCacheMinFiles)
	danglingSelector := []byte("package p\nfunc _() {\n\tx.\n}")
	files = append(files, makeFakeFileHandle("file:///bad1", danglingSelector))
	files = append(files, makeFakeFileHandle("file:///bad2", danglingSelector))

	// Parsing should succeed even though we overflow the padding.
	cache := newParseCache(0)
	_, err := cache.parseFiles(context.Background(), token.NewFileSet(), source.ParseFull, false, files...)
	if err != nil {
		t.Fatal(err)
	}
}

// Re-parsing the first file should not panic.
func TestParseCache_Issue59097(t *testing.T) {
	skipIfNoParseCache(t)

	defer func(padding int) {
		parsePadding = padding
	}(parsePadding)
	parsePadding = 0

	danglingSelector := []byte("package p\nfunc _() {\n\tx.\n}")
	files := []source.FileHandle{makeFakeFileHandle("file:///bad", danglingSelector)}

	// Parsing should succeed even though we overflow the padding.
	cache := newParseCache(0)
	_, err := cache.parseFiles(context.Background(), token.NewFileSet(), source.ParseFull, false, files...)
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseCache_TimeEviction(t *testing.T) {
	skipIfNoParseCache(t)

	ctx := context.Background()
	fset := token.NewFileSet()
	uri := span.URI("file:///myfile")
	fh := makeFakeFileHandle(uri, []byte("package p\n\nconst _ = \"foo\""))

	const gcDuration = 10 * time.Millisecond
	cache := newParseCache(gcDuration)
	cache.stop() // we'll manage GC manually, for testing.

	pgfs0, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh, fh)
	if err != nil {
		t.Fatal(err)
	}

	files := dummyFileHandles(parseCacheMinFiles)
	_, err = cache.parseFiles(ctx, fset, source.ParseFull, false, files...)
	if err != nil {
		t.Fatal(err)
	}

	// Even after filling up the 'min' files, we get a cache hit for our original file.
	pgfs1, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh, fh)
	if err != nil {
		t.Fatal(err)
	}

	if pgfs0[0] != pgfs1[0] {
		t.Errorf("before GC, got unexpected cache miss")
	}

	// But after GC, we get a cache miss.
	_, err = cache.parseFiles(ctx, fset, source.ParseFull, false, files...) // mark dummy files as newer
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(gcDuration)
	cache.gcOnce()

	pgfs2, err := cache.parseFiles(ctx, fset, source.ParseFull, false, fh, fh)
	if err != nil {
		t.Fatal(err)
	}

	if pgfs0[0] == pgfs2[0] {
		t.Errorf("after GC, got unexpected cache hit for %s", pgfs0[0].URI)
	}
}

func TestParseCache_Duplicates(t *testing.T) {
	skipIfNoParseCache(t)

	ctx := context.Background()
	uri := span.URI("file:///myfile")
	fh := makeFakeFileHandle(uri, []byte("package p\n\nconst _ = \"foo\""))

	cache := newParseCache(0)
	pgfs, err := cache.parseFiles(ctx, token.NewFileSet(), source.ParseFull, false, fh, fh)
	if err != nil {
		t.Fatal(err)
	}
	if pgfs[0] != pgfs[1] {
		t.Errorf("parseFiles(fh, fh): = [%p, %p], want duplicate files", pgfs[0].File, pgfs[1].File)
	}
}

func dummyFileHandles(n int) []source.FileHandle {
	var fhs []source.FileHandle
	for i := 0; i < n; i++ {
		uri := span.URI(fmt.Sprintf("file:///_%d", i))
		src := []byte(fmt.Sprintf("package p\nvar _ = %d", i))
		fhs = append(fhs, makeFakeFileHandle(uri, src))
	}
	return fhs
}

func makeFakeFileHandle(uri span.URI, src []byte) fakeFileHandle {
	return fakeFileHandle{
		uri:  uri,
		data: src,
		hash: source.HashOf(src),
	}
}

type fakeFileHandle struct {
	source.FileHandle
	uri  span.URI
	data []byte
	hash source.Hash
}

func (h fakeFileHandle) URI() span.URI {
	return h.uri
}

func (h fakeFileHandle) Content() ([]byte, error) {
	return h.data, nil
}

func (h fakeFileHandle) FileIdentity() source.FileIdentity {
	return source.FileIdentity{
		URI:  h.uri,
		Hash: h.hash,
	}
}
