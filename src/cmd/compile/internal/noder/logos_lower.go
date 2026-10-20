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
// eject, plus the compile arguments of stashed files for next reports,
// plus the pending eject edits of a changed package.
var logosStore = struct {
	sync.Mutex
	src   map[*syntax.File][]byte
	names map[*syntax.File]string
	edits map[*syntax.File][]logos.Edit
}{src: make(map[*syntax.File][]byte), names: make(map[*syntax.File]string)}

// logosStashesGo reports whether Go files keep their original bytes: while a
// next sidecar is armed, while logos lint reads their comments, and in hints
// mode, which converts positions from the bytes the compiler parsed.
func logosStashesGo() bool {
	return base.LogosNextOut != "" || base.LogosLintOnly || base.LogosMode == "hints"
}

// parseLogosFile parses one source file, remembering the original bytes
// of logos files for lowering and eject. Go files are remembered too
// when logosStashesGo reports it.
func parseLogosFile(fbase *syntax.PosBase, f *os.File, filename string, errh syntax.ErrorHandler, pragh syntax.PragmaHandler) *syntax.File {
	mode := syntax.CheckBranches | logosMode(filename)
	if !isLogosFile(filename) && !logosStashesGo() {
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
		logosStore.names[file] = filename
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

// logosRecheckHook runs the post-check loGo/s rewrite for the package that
// conf checked: lowering for a loGo/s package, MustNonNil wraps for a Go
// package that calls loGo/s code (M3-S7). It reports whether the caller
// must recheck.
func logosRecheckHook(conf *types2.Config, pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	logosHintsBegin(pkg, info, files)
	if conf.Logos {
		return lowerLogosHook(pkg, info, files)
	}
	return logosGoCallerHook(pkg, info, files)
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
	logosHintsLowered(files, edits)
	// M2-S7: rewrite value-used unions to methods-only eject text. The
	// native tree is untouched, so this never forces a recheck by
	// itself; it only fails an eject on a dual-use union, with the
	// command prefix added once here.
	if base.LogosEject != "" {
		unionEdits, err := logos.EjectUnionEdits(info, files, src)
		if err != nil {
			base.Fatalf("logos eject: %v", err)
		}
		for file, list := range unionEdits {
			edits[file] = append(edits[file], list...)
		}
	} else {
		for file, list := range logos.NativeUnionEdits(info, files, src) {
			edits[file] = append(edits[file], list...)
		}
	}
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
