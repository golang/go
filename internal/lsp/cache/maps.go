// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/persistent"
	"golang.org/x/tools/internal/span"
)

// TODO(euroelessar): Use generics once support for go1.17 is dropped.

type filesMap struct {
	impl *persistent.Map
}

func newFilesMap() filesMap {
	return filesMap{
		impl: persistent.NewMap(func(a, b interface{}) bool {
			return a.(span.URI) < b.(span.URI)
		}),
	}
}

func (m filesMap) Clone() filesMap {
	return filesMap{
		impl: m.impl.Clone(),
	}
}

func (m filesMap) Destroy() {
	m.impl.Destroy()
}

func (m filesMap) Get(key span.URI) (source.VersionedFileHandle, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return nil, false
	}
	return value.(source.VersionedFileHandle), true
}

func (m filesMap) Range(do func(key span.URI, value source.VersionedFileHandle)) {
	m.impl.Range(func(key, value interface{}) {
		do(key.(span.URI), value.(source.VersionedFileHandle))
	})
}

func (m filesMap) Set(key span.URI, value source.VersionedFileHandle) {
	m.impl.Set(key, value, nil)
}

func (m filesMap) Delete(key span.URI) {
	m.impl.Delete(key)
}

type goFilesMap struct {
	impl *persistent.Map
}

func newGoFilesMap() goFilesMap {
	return goFilesMap{
		impl: persistent.NewMap(func(a, b interface{}) bool {
			return parseKeyLess(a.(parseKey), b.(parseKey))
		}),
	}
}

func parseKeyLess(a, b parseKey) bool {
	if a.mode != b.mode {
		return a.mode < b.mode
	}
	if a.file.Hash != b.file.Hash {
		return a.file.Hash.Less(b.file.Hash)
	}
	return a.file.URI < b.file.URI
}

func (m goFilesMap) Clone() goFilesMap {
	return goFilesMap{
		impl: m.impl.Clone(),
	}
}

func (m goFilesMap) Destroy() {
	m.impl.Destroy()
}

func (m goFilesMap) Get(key parseKey) (*parseGoHandle, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return nil, false
	}
	return value.(*parseGoHandle), true
}

func (m goFilesMap) Range(do func(key parseKey, value *parseGoHandle)) {
	m.impl.Range(func(key, value interface{}) {
		do(key.(parseKey), value.(*parseGoHandle))
	})
}

func (m goFilesMap) Set(key parseKey, value *parseGoHandle, release func()) {
	m.impl.Set(key, value, func(key, value interface{}) {
		release()
	})
}

func (m goFilesMap) Delete(key parseKey) {
	m.impl.Delete(key)
}

type parseKeysByURIMap struct {
	impl *persistent.Map
}

func newParseKeysByURIMap() parseKeysByURIMap {
	return parseKeysByURIMap{
		impl: persistent.NewMap(func(a, b interface{}) bool {
			return a.(span.URI) < b.(span.URI)
		}),
	}
}

func (m parseKeysByURIMap) Clone() parseKeysByURIMap {
	return parseKeysByURIMap{
		impl: m.impl.Clone(),
	}
}

func (m parseKeysByURIMap) Destroy() {
	m.impl.Destroy()
}

func (m parseKeysByURIMap) Get(key span.URI) ([]parseKey, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return nil, false
	}
	return value.([]parseKey), true
}

func (m parseKeysByURIMap) Range(do func(key span.URI, value []parseKey)) {
	m.impl.Range(func(key, value interface{}) {
		do(key.(span.URI), value.([]parseKey))
	})
}

func (m parseKeysByURIMap) Set(key span.URI, value []parseKey) {
	m.impl.Set(key, value, nil)
}

func (m parseKeysByURIMap) Delete(key span.URI) {
	m.impl.Delete(key)
}
