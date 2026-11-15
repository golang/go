// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"internal/logosfile"
	"io"
	"maps"
	"os"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

var logosStore = struct {
	sync.Mutex
	src   map[*syntax.File][]byte
	cgo   map[*syntax.File][]byte
	names map[*syntax.File]string
	edits map[*syntax.File][]logos.Edit
}{src: make(map[*syntax.File][]byte), cgo: make(map[*syntax.File][]byte), names: make(map[*syntax.File]string)}

func logosStashesGo() bool {
	return base.LogosNextOut != "" || base.LogosLintEnabled() || base.LogosMode == "hints"
}

func parseLogosFile(fbase *syntax.PosBase, f *os.File, filename string, errh syntax.ErrorHandler, pragh syntax.PragmaHandler) *syntax.File {
	mode := syntax.CheckBranches | logosMode(filename)
	cgoUser := isLogosCgoUser(filename)
	if !logosfile.Is(filename) && !logosStashesGo() && !cgoUser {
		file, _ := syntax.Parse(fbase, f, errh, pragh, mode)
		return file
	}
	data, err := io.ReadAll(f)
	if err != nil {
		errh(syntax.Error{Msg: err.Error()})
		return nil
	}
	if logosfile.Is(filename) {
		data = rewriteLogosWasmDirectives(data, logosMode(filename))
	}
	file, _ := syntax.Parse(fbase, bytes.NewReader(data), errh, pragh, mode)
	if file != nil {
		logosStore.Lock()
		defer logosStore.Unlock()
		if logosfile.Is(filename) || logosStashesGo() {
			logosStore.src[file] = data
			logosStore.names[file] = filename
		}
		if cgoUser {
			logosStore.cgo[file] = data
		}
	}
	return file
}

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

func logosCgoSources(files []*syntax.File) map[*syntax.File][]byte {
	src := make(map[*syntax.File][]byte)
	logosStore.Lock()
	defer logosStore.Unlock()
	for _, file := range files {
		if data, ok := logosStore.cgo[file]; ok {
			src[file] = data
			delete(logosStore.cgo, file)
		}
	}
	return src
}

func logosProvenReadonlyMap(pkg *types2.Package) map[string]bool {
	handoff := make(map[string]bool)
	if pkg == nil || pkg.Scope() == nil {
		return handoff
	}
	for _, name := range pkg.Scope().Names() {
		obj := pkg.Scope().Lookup(name)
		switch o := obj.(type) {
		case *types2.Func:
			if fn := o.Origin(); fn != nil && fn.LogosReadonly() {
				if k, ok := types2.LogosProvenKey(fn); ok {
					handoff[k] = true
				}
			}
		case *types2.TypeName:
			if n, ok := o.Type().(*types2.Named); ok {
				n = n.Origin()
				for i := range n.NumMethods() {
					if fn := n.Method(i).Origin(); fn != nil && fn.LogosReadonly() {
						if k, ok := types2.LogosProvenKey(fn); ok {
							handoff[k] = true
						}
					}
				}
			}
		}
	}
	return handoff
}

func logosWriteCheckedFacts(pkg *types2.Package, info *types2.Info, files []*syntax.File) {
	if pkg == nil {
		return
	}
	f := facts.WriteChecked(pkg, info, files, facts.Current())
	if base.LogosGoFacts && !base.Logos {
		f = facts.EmitGoMutations(pkg, info, files, f)
	}
	if f == nil {
		return
	}
	if !base.Logos {
		f.GoPackage = true
	}
	facts.SetCurrent(f)
}

func logosPublishConstViewDecls(decls []syntax.Pos) {
	seen := make(map[cvDeclKey]bool, len(decls))
	for _, pos := range decls {
		if !pos.IsKnown() || pos.Base() == nil {
			continue
		}
		seen[cvDeclKey{trimFilename(pos.Base()), pos.Line(), pos.Col()}] = true
	}
	escape.LogosConstViewDecl = func(xpos src.XPos) bool {
		p := base.Ctxt.PosTable.Pos(xpos)
		b := p.Base()
		if !p.IsKnown() || b == nil {
			return false
		}
		return seen[cvDeclKey{b.AbsFilename(), p.Line(), p.Col()}]
	}
}

type cvDeclKey struct {
	file string
	line uint
	col  uint
}

func logosRecheckHook(conf *types2.Config, pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	logosHintsBegin(pkg, info, files)
	if conf.Logos {
		types2.LogosSealExportedReadonly(pkg, info, files)
		conf.LogosProvenReadonly = logosProvenReadonlyMap(pkg)
		mustRecheck := lowerLogosHook(pkg, info, files)
		return mustRecheck
	}
	logosCgoSources(files)
	return logosGoCallerHook(pkg, info, files)
}

func lowerLogosHook(pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	src := logosSources(files)
	lowerSrc := maps.Clone(src)
	maps.Copy(lowerSrc, logosCgoSources(files))
	if len(lowerSrc) == 0 {
		return false
	}
	changed, edits := logos.Lower(pkg, info, files, lowerSrc)
	logosPublishConstViewDecls(logos.ConstViewDecls())
	if changed {
		logos.LogosRestoreErasedAnonReadonly()
	} else {
		logos.LogosClearErasedAnonReadonly()
	}
	for file := range edits {
		if _, ok := src[file]; !ok {
			delete(edits, file)
		}
	}
	logosHintsLowered(files, edits)
	if base.LogosEject != "" {
		unionEdits, err := logos.EjectUnionEdits(info, files, src)
		if err != nil {
			base.Fatalf("logos eject: %v", err)
		}
		importEdits := logos.UnionImportEdits(info, src, edits, unionEdits)
		for file, list := range unionEdits {
			edits[file] = append(edits[file], list...)
		}
		for file, list := range importEdits {
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

func logosConfigureOwnedConstantsLowered(conf *types2.Config) {
	conf.LogosOwnedTypes = logos.OwnedTypeNodes()
}

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
