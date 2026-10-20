// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"maps"
	"path/filepath"
	"slices"
)

// refresh recompiles the dirty packages and brings gopls and the client up
// to date with the result.
func (p *Proxy) refresh(ctx context.Context) {
	p.work.Lock()
	defer p.work.Unlock()
	p.refreshLocked(ctx)
}

// refreshLocked is refresh for a caller that holds p.work. It compiles
// every dirty package, hands the lowered documents to gopls, and publishes
// diagnostics. It does nothing once the proxy is stopping, so a late
// background run cannot use a compiler that stop is about to close.
func (p *Proxy) refreshLocked(ctx context.Context) {
	dirs, buffers, versions := p.takeDirty()
	for _, dir := range dirs {
		low, diags := p.compile(ctx, dir, buffers)
		p.apply(dir, low, diags, versions)
	}
	if len(dirs) > 0 {
		p.syncGopls()
		p.publishAll()
	}
}

// takeDirty marks every dirty package clean and returns their directories,
// the texts of the open documents by path, and the version of each. A
// stopped proxy has nothing to compile.
func (p *Proxy) takeDirty() ([]string, map[string][]byte, map[string]int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return nil, nil, nil
	}
	var dirs []string
	for _, dir := range slices.Sorted(maps.Keys(p.units)) {
		if u := p.units[dir]; u.dirty {
			u.dirty = false
			dirs = append(dirs, dir)
		}
	}
	buffers := make(map[string][]byte, len(p.docs))
	versions := make(map[string]int, len(p.docs))
	for path, doc := range p.docs {
		buffers[path] = doc.text.data
		versions[path] = doc.version
	}
	return dirs, buffers, versions
}

// compile lowers the package in dir and collects its diagnostics. A
// failed run yields a nil lowering and no diagnostics; the log has the
// reason.
func (p *Proxy) compile(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, []Diagnostic) {
	low, err := p.opts.Compiler.Lower(ctx, dir, buffers)
	if err != nil {
		p.logf("lowering %s: %v", dir, err)
		return nil, nil
	}
	diags, err := p.opts.Compiler.Diagnose(ctx, dir, buffers)
	if err != nil {
		p.logf("diagnosing %s: %v", dir, err)
	}
	return low, diags
}

// apply installs the result of compiling the package in dir. A nil
// lowering means the run failed: the package keeps its closure, and every
// file of it turns stale.
func (p *Proxy) apply(dir string, low *Lowering, diags []Diagnostic, versions map[string]int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	u := p.units[dir]
	if u == nil {
		return
	}
	if low == nil {
		low = &Lowering{Dirs: slices.Collect(maps.Keys(u.dirs)), Missing: slices.Collect(maps.Keys(u.paths))}
	}
	u.dirs = setOf(low.Dirs)
	u.diags = diags
	u.diagVersions = versions
	u.hintsOK = false
	u.gen++
	u.paths = make(map[string]bool, len(low.Files)+len(low.Missing))
	for _, f := range low.Files {
		p.installLocked(f)
		u.paths[f.Path] = true
	}
	for _, path := range low.Missing {
		p.staleLocked(path)
		u.paths[path] = true
	}
	p.pruneLocked()
}

// installLocked records the lowering of one file. An edit map that does
// not fit its texts makes the file stale, like a file that failed to
// compile.
func (p *Proxy) installLocked(f LoweredFile) {
	pm, err := newPosMap(f.Src, f.Out, f.Map)
	if err != nil {
		p.logf("%v", err)
		p.staleLocked(f.Path)
		return
	}
	doc := p.docs[f.Path]
	p.files[f.Path] = &lowered{
		src:   f.Path,
		virt:  virtualPath(f.Path),
		pm:    pm,
		fresh: doc == nil || bytes.Equal(doc.text.data, f.Src),
	}
}

// staleLocked marks the lowering of path, if any, as no longer matching
// the source.
func (p *Proxy) staleLocked(path string) {
	if lw := p.files[path]; lw != nil && lw.fresh {
		stale := *lw
		stale.fresh = false
		p.files[path] = &stale
	}
}

// pruneLocked drops the lowerings no unit needs any more, and publishes
// the new tables.
func (p *Proxy) pruneLocked() {
	need := make(map[string]bool)
	for _, u := range p.units {
		maps.Copy(need, u.paths)
	}
	maps.DeleteFunc(p.files, func(path string, _ *lowered) bool { return !need[path] })
	tb := &tables{bySrc: make(map[string]*lowered, len(p.files)), byVirt: make(map[string]*lowered, len(p.files))}
	for path, lw := range p.files {
		tb.bySrc[path] = lw
		tb.byVirt[lw.virt] = lw
	}
	p.tables.Store(tb)
}

// setOf returns the set of the elements of list.
func setOf(list []string) map[string]bool {
	set := make(map[string]bool, len(list))
	for _, s := range list {
		set[s] = true
	}
	return set
}

// syncGopls makes the lowered documents open in gopls match the
// lowerings: it opens the new ones, updates the changed ones, and closes
// the ones no unit needs.
func (p *Proxy) syncGopls() {
	for _, m := range p.virtualMessages() {
		p.send(p.server, m)
	}
}

// virtualMessages computes the messages that bring gopls in line with the
// lowerings, and records what it will then hold.
func (p *Proxy) virtualMessages() []*message {
	p.mu.Lock()
	defer p.mu.Unlock()
	var msgs []*message
	want := make(map[string]bool, len(p.files))
	for _, path := range slices.Sorted(maps.Keys(p.files)) {
		lw := p.files[path]
		want[lw.virt] = true
		msgs = append(msgs, p.openOrChangeLocked(lw)...)
	}
	for _, virt := range slices.Sorted(maps.Keys(p.virt)) {
		if !want[virt] {
			msgs = append(msgs, p.notification("textDocument/didClose", docParams{TextDocument: docIdentifier{URI: pathURI(virt)}})...)
			delete(p.virt, virt)
			delete(p.srvDiags, pathURI(virt))
		}
	}
	return msgs
}

// openOrChangeLocked returns the notification that gives gopls the lowered
// text of lw, or none when it holds that text already.
func (p *Proxy) openOrChangeLocked(lw *lowered) []*message {
	out := lw.pm.low.data
	vd := p.virt[lw.virt]
	if vd == nil {
		p.virt[lw.virt] = &virtDoc{lw: lw, version: 1}
		return p.notification("textDocument/didOpen", didOpenParams{
			TextDocument: textDocumentItem{URI: lw.virtURI(), LanguageID: "go", Version: 1, Text: string(out)},
		})
	}
	same := bytes.Equal(vd.lw.pm.low.data, out)
	vd.lw = lw
	if same {
		return nil
	}
	vd.version++
	return p.notification("textDocument/didChange", didChangeParams{
		TextDocument:   versionedID{URI: lw.virtURI(), Version: vd.version},
		ContentChanges: []contentChange{{Text: string(out)}},
	})
}

// notification returns the notification message of method with params, as
// a list of one message, or of none when the params do not encode.
func (p *Proxy) notification(method string, params any) []*message {
	raw, err := json.Marshal(params)
	if err != nil {
		p.logf("encoding %s: %v", method, err)
		return nil
	}
	return []*message{{Method: method, Params: raw}}
}

// unitFor returns the unit of the package that holds path. The caller
// holds p.mu.
func (p *Proxy) unitFor(path string) *unit {
	return p.units[filepath.Dir(path)]
}
