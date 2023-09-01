// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/persistent"
)

type filesMap struct {
	impl       *persistent.Map[span.URI, source.FileHandle]
	overlayMap map[span.URI]*Overlay // the subset that are overlays
}

func newFilesMap() filesMap {
	return filesMap{
		impl:       new(persistent.Map[span.URI, source.FileHandle]),
		overlayMap: make(map[span.URI]*Overlay),
	}
}

func (m filesMap) Clone() filesMap {
	overlays := make(map[span.URI]*Overlay, len(m.overlayMap))
	for k, v := range m.overlayMap {
		overlays[k] = v
	}
	return filesMap{
		impl:       m.impl.Clone(),
		overlayMap: overlays,
	}
}

func (m filesMap) Destroy() {
	m.impl.Destroy()
}

func (m filesMap) Get(key span.URI) (source.FileHandle, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return nil, false
	}
	return value.(source.FileHandle), true
}

func (m filesMap) Range(do func(key span.URI, value source.FileHandle)) {
	m.impl.Range(do)
}

func (m filesMap) Set(key span.URI, value source.FileHandle) {
	m.impl.Set(key, value, nil)

	if o, ok := value.(*Overlay); ok {
		m.overlayMap[key] = o
	} else {
		// Setting a non-overlay must delete the corresponding overlay, to preserve
		// the accuracy of the overlay set.
		delete(m.overlayMap, key)
	}
}

func (m *filesMap) Delete(key span.URI) {
	m.impl.Delete(key)
	delete(m.overlayMap, key)
}

// overlays returns a new unordered array of overlay files.
func (m filesMap) overlays() []*Overlay {
	// In practice we will always have at least one overlay, so there is no need
	// to optimize for the len=0 case by returning a nil slice.
	overlays := make([]*Overlay, 0, len(m.overlayMap))
	for _, o := range m.overlayMap {
		overlays = append(overlays, o)
	}
	return overlays
}
