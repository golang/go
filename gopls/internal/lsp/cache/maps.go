// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/persistent"
)

// TODO(euroelessar): Use generics once support for go1.17 is dropped.

type filesMap struct {
	impl *persistent.Map
}

// uriLessInterface is the < relation for "any" values containing span.URIs.
func uriLessInterface(a, b interface{}) bool {
	return a.(span.URI) < b.(span.URI)
}

func newFilesMap() filesMap {
	return filesMap{
		impl: persistent.NewMap(uriLessInterface),
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

func (m filesMap) Get(key span.URI) (source.FileHandle, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return nil, false
	}
	return value.(source.FileHandle), true
}

func (m filesMap) Range(do func(key span.URI, value source.FileHandle)) {
	m.impl.Range(func(key, value interface{}) {
		do(key.(span.URI), value.(source.FileHandle))
	})
}

func (m filesMap) Set(key span.URI, value source.FileHandle) {
	m.impl.Set(key, value, nil)
}

func (m filesMap) Delete(key span.URI) {
	m.impl.Delete(key)
}

func packageIDLessInterface(x, y interface{}) bool {
	return x.(PackageID) < y.(PackageID)
}

type knownDirsSet struct {
	impl *persistent.Map
}

func newKnownDirsSet() knownDirsSet {
	return knownDirsSet{
		impl: persistent.NewMap(func(a, b interface{}) bool {
			return a.(span.URI) < b.(span.URI)
		}),
	}
}

func (s knownDirsSet) Clone() knownDirsSet {
	return knownDirsSet{
		impl: s.impl.Clone(),
	}
}

func (s knownDirsSet) Destroy() {
	s.impl.Destroy()
}

func (s knownDirsSet) Contains(key span.URI) bool {
	_, ok := s.impl.Get(key)
	return ok
}

func (s knownDirsSet) Range(do func(key span.URI)) {
	s.impl.Range(func(key, value interface{}) {
		do(key.(span.URI))
	})
}

func (s knownDirsSet) SetAll(other knownDirsSet) {
	s.impl.SetAll(other.impl)
}

func (s knownDirsSet) Insert(key span.URI) {
	s.impl.Set(key, nil, nil)
}

func (s knownDirsSet) Remove(key span.URI) {
	s.impl.Delete(key)
}

// analysisKeyLessInterface is the less-than relation for analysisKey
// values wrapped in an interface.
func analysisKeyLessInterface(a, b interface{}) bool {
	x, y := a.(analysisKey), b.(analysisKey)
	if cmp := strings.Compare(x.analyzerNames, y.analyzerNames); cmp != 0 {
		return cmp < 0
	}
	return x.pkgid < y.pkgid
}
