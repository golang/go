// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sync"
)

type document struct {
	uri     string
	version int
	text    *text
}

type unit struct {
	dir   string
	dirs  map[string]bool
	paths map[string]bool
	diags []Diagnostic
	dirty bool

	diagVersions map[string]int

	gen int

	hints   json.RawMessage
	hintsOK bool
}

type store struct {
	mu      sync.Mutex
	docs    map[string]*document
	units   map[string]*unit
	flights map[string]chan struct{}
}

func (s *store) init() {
	s.docs = make(map[string]*document)
	s.units = make(map[string]*unit)
	s.flights = make(map[string]chan struct{})
}

func (s *store) open(path string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.docs[path] != nil
}

func (s *store) setDoc(uri, path string, version int, t *text) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs[path] = &document{uri: uri, version: version, text: t}
	s.touchLocked(path)
}

func (s *store) change(path string, version int, changes []contentChange) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	doc := s.docs[path]
	if doc == nil {
		return false
	}
	if version <= doc.version {
		return false
	}
	next := applyChanges(doc.text, changes)
	if len(next.data) > maxDocBytes {
		return false
	}
	doc.text = next
	doc.version = version
	s.touchLocked(path)
	return true
}

func (s *store) forget(path string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	doc := s.docs[path]
	if doc == nil {
		return false
	}
	unsaved := !diskEqual(path, doc.text.data)
	delete(s.docs, path)
	s.touchLocked(path)
	s.dropUnitLocked(filepath.Dir(path))
	return unsaved
}

func diskEqual(path string, data []byte) bool {
	disk, err := os.ReadFile(path)
	return err == nil && bytes.Equal(disk, data)
}

func (s *store) touchLocked(path string) {
	dir := filepath.Dir(path)
	if s.units[dir] == nil {
		s.units[dir] = &unit{dir: dir}
	}
	for _, u := range s.units {
		if u.dir == dir || u.dirs[dir] {
			u.dirty = true
			u.gen++
			u.hintsOK = false
		}
	}
}

func (s *store) dropUnitLocked(dir string) {
	for path := range s.docs {
		if filepath.Dir(path) == dir {
			return
		}
	}
	delete(s.units, dir)
}

func (s *store) unitFor(path string) *unit {
	return s.units[filepath.Dir(path)]
}

func (s *store) unitNeed() map[string]bool {
	need := make(map[string]bool)
	for _, u := range s.units {
		maps.Copy(need, u.paths)
	}
	return need
}

func (p *Proxy) ensure(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		p.st.mu.Lock()
		var dir string
		for _, d := range slices.Sorted(maps.Keys(p.st.units)) {
			if u := p.st.units[d]; u.dirty {
				dir = d
				break
			}
		}
		if dir == "" {
			p.st.mu.Unlock()
			return
		}
		if done, ok := p.st.flights[dir]; ok {
			p.st.mu.Unlock()
			select {
			case <-done:
			case <-ctx.Done():
				return
			}
			continue
		}
		done := make(chan struct{})
		p.st.flights[dir] = done
		p.st.mu.Unlock()

		p.compileUnit(ctx, dir)

		p.st.mu.Lock()
		delete(p.st.flights, dir)
		p.st.mu.Unlock()
		close(done)
	}
}

func (p *Proxy) compileUnit(ctx context.Context, dir string) {
	buffers, versions := p.takeDirty(dir)
	if buffers == nil {
		return
	}
	low, diags, lerr := p.compile(ctx, dir, buffers)
	p.recordLowering(dir, lerr)
	p.apply(dir, low, diags, versions)
	p.syncGopls()
	p.publishAll()
}

func (p *Proxy) recordLowering(dir string, lerr error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if lerr != nil {
		if len(p.lowerErr) >= maxLowerErrs {
			clear(p.lowerErr)
		}
		p.lowerErr[dir] = lerr
	} else {
		delete(p.lowerErr, dir)
	}
}

func (p *Proxy) takeDirty(dir string) (map[string][]byte, map[string]int) {
	p.stopMu.Lock()
	stopped := p.stopped
	p.stopMu.Unlock()
	if stopped {
		return nil, nil
	}
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	u := p.st.units[dir]
	if u == nil || !u.dirty {
		return nil, nil
	}
	u.dirty = false
	buffers := make(map[string][]byte, len(p.st.docs))
	versions := make(map[string]int, len(p.st.docs))
	for path, doc := range p.st.docs {
		buffers[path] = doc.text.data
		versions[path] = doc.version
	}
	return buffers, versions
}

func (p *Proxy) compile(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, []Diagnostic, error) {
	low, err := p.opts.Compiler.Lower(ctx, dir, buffers)
	if err != nil {
		p.logf("lowering %s: %v", dir, err)
		return nil, nil, err
	}
	diags, err := p.opts.Compiler.Diagnose(ctx, dir, buffers)
	if err != nil {
		p.logf("diagnosing %s: %v", dir, err)
	}
	return low, diags, nil
}

func (p *Proxy) apply(dir string, low *Lowering, diags []Diagnostic, versions map[string]int) {
	p.st.mu.Lock()
	u := p.st.units[dir]
	if u == nil {
		p.st.mu.Unlock()
		return
	}
	if low == nil {
		low = &Lowering{Dirs: slices.Collect(maps.Keys(u.dirs)), Missing: slices.Collect(maps.Keys(u.paths))}
	}
	u.dirs = make(map[string]bool, len(low.Dirs))
	for _, d := range low.Dirs {
		u.dirs[d] = true
	}
	u.diags = diags
	u.diagVersions = versions
	u.hintsOK = false
	u.gen++
	u.paths = make(map[string]bool, len(low.Files)+len(low.Missing))
	fresh := make(map[string]*lowered, len(low.Files))
	for _, f := range low.Files {
		u.paths[f.Path] = true
		pm, err := newPosMap(f.Src, f.Out, f.Map)
		if err != nil {
			p.logf("%v", err)
			fresh[f.Path] = nil
			continue
		}
		doc := p.st.docs[f.Path]
		fresh[f.Path] = &lowered{
			src:   f.Path,
			virt:  virtualPath(f.Path),
			pm:    pm,
			fresh: doc == nil || bytes.Equal(doc.text.data, f.Src),
			gen:   u.gen,
		}
	}
	for _, path := range low.Missing {
		u.paths[path] = true
	}
	need := p.st.unitNeed()
	p.st.mu.Unlock()

	p.mu.Lock()
	defer p.mu.Unlock()
	for path, lw := range fresh {
		if lw == nil {
			p.staleLocked(path)
			continue
		}
		p.files[path] = lw
	}
	for _, path := range low.Missing {
		p.staleLocked(path)
	}
	p.pruneLocked(need)
}

type snapshotCountsResult struct {
	Dirs      map[string]int `json:"dirs"`
	DiagCache int            `json:"diagCache"`
}

func (p *Proxy) snapshotCounts(ctx context.Context, m *message) {
	p.ensure(ctx)
	p.respond(m.ID, p.countSnapshots())
}

func (p *Proxy) countSnapshots() snapshotCountsResult {
	p.work.Lock()
	defer p.work.Unlock()
	p.mu.Lock()
	defer p.mu.Unlock()
	gens := make(map[string]map[int]struct{})
	for path, lw := range p.files {
		dir := filepath.Dir(path)
		set := gens[dir]
		if set == nil {
			set = make(map[int]struct{})
			gens[dir] = set
		}
		set[lw.gen] = struct{}{}
	}
	out := make(map[string]int, len(gens))
	for dir, set := range gens {
		out[dir] = len(set)
	}
	return snapshotCountsResult{Dirs: out, DiagCache: len(p.srvDiags)}
}

func (p *Proxy) staleLocked(path string) {
	if lw := p.files[path]; lw != nil && lw.fresh {
		stale := *lw
		stale.fresh = false
		p.files[path] = &stale
	}
}

func (p *Proxy) pruneLocked(need map[string]bool) {
	maps.DeleteFunc(p.files, func(path string, _ *lowered) bool { return !need[path] })
	tb := &tables{bySrc: make(map[string]*lowered, len(p.files)), byVirt: make(map[string]*lowered, len(p.files))}
	for path, lw := range p.files {
		tb.bySrc[path] = lw
		tb.byVirt[lw.virt] = lw
	}
	p.tables.Store(tb)
}

func (p *Proxy) syncGopls() {
	for _, m := range p.virtualMessages() {
		p.send(p.server, m)
	}
}

func (p *Proxy) virtualMessages() []*message {
	p.st.mu.Lock()
	p.mu.Lock()
	defer p.st.mu.Unlock()
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

func (p *Proxy) notification(method string, params any) []*message {
	raw, err := json.Marshal(params)
	if err != nil {
		p.logf("encoding %s: %v", method, err)
		return nil
	}
	return []*message{{Method: method, Params: raw}}
}
