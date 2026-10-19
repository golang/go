// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"io"
	"os"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

// This file holds the loGo/s lowering integration for the compiler
// frontend. Upstream files carry only hook lines; all logic lives here.
// Lowering runs once after the first successful types2 check, then the
// lowered tree is rechecked as ordinary Go.

// logosStore keeps the original bytes of logos files for lowering and
// eject, plus the pending eject edits of a changed package.
var logosStore = struct {
	sync.Mutex
	src   map[*syntax.File][]byte
	edits map[*syntax.File][]logos.Edit
}{src: make(map[*syntax.File][]byte)}

// parseLogosFile parses one source file, remembering the original bytes
// of logos files for lowering and eject.
func parseLogosFile(fbase *syntax.PosBase, f *os.File, filename string, errh syntax.ErrorHandler, pragh syntax.PragmaHandler) *syntax.File {
	mode := syntax.CheckBranches | logosMode(filename)
	if !isLogosFile(filename) {
		file, _ := syntax.Parse(fbase, f, errh, pragh, mode)
		return file
	}
	data, err := io.ReadAll(f)
	if err != nil {
		errh(syntax.Error{Msg: err.Error()})
		return nil
	}
	file, _ := syntax.Parse(fbase, bytes.NewReader(data), errh, pragh, mode)
	if file != nil {
		logosStore.Lock()
		defer logosStore.Unlock()
		logosStore.src[file] = data
	}
	return file
}

// logosSources collects the remembered bytes of files, consuming them.
func logosSources(files []*syntax.File) map[*syntax.File][]byte {
	src := make(map[*syntax.File][]byte, len(files))
	logosStore.Lock()
	defer logosStore.Unlock()
	for _, file := range files {
		if data, ok := logosStore.src[file]; ok {
			src[file] = data
			delete(logosStore.src, file)
		}
	}
	return src
}

// lowerLogosHook lowers logos nodes after the first check and resets type
// checking for the recheck. It reports whether the caller must recheck.
// Sources are stashed for eject even when nothing changed, so unchanged
// logos files still eject as copies.
func lowerLogosHook(pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	src := logosSources(files)
	if len(src) == 0 {
		return false
	}
	changed, edits := logos.Lower(pkg, info, files, src)
	logosStore.Lock()
	defer logosStore.Unlock()
	for file, data := range src {
		logosStore.src[file] = data
	}
	logosStore.edits = edits
	if !changed {
		return false
	}
	for _, p := range types.PkgMap() {
		p.Direct = false
	}
	typecheck.Target.Imports = nil
	return true
}

// writeLogosEject writes ejected Go sources when -logos.eject is set.
func writeLogosEject(files []*syntax.File) {
	dir := base.LogosEject
	if dir == "" {
		return
	}
	logosStore.Lock()
	src := logosStore.src
	edits := logosStore.edits
	logosStore.src = make(map[*syntax.File][]byte)
	logosStore.edits = nil
	logosStore.Unlock()
	if len(src) == 0 {
		return
	}
	if err := logos.Eject(dir, files, src, edits); err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}
