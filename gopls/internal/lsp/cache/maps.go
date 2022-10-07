// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
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

func parseKeyLessInterface(a, b interface{}) bool {
	return parseKeyLess(a.(parseKey), b.(parseKey))
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

type isActivePackageCacheMap struct {
	impl *persistent.Map
}

func newIsActivePackageCacheMap() isActivePackageCacheMap {
	return isActivePackageCacheMap{
		impl: persistent.NewMap(func(a, b interface{}) bool {
			return a.(PackageID) < b.(PackageID)
		}),
	}
}

func (m isActivePackageCacheMap) Clone() isActivePackageCacheMap {
	return isActivePackageCacheMap{
		impl: m.impl.Clone(),
	}
}

func (m isActivePackageCacheMap) Destroy() {
	m.impl.Destroy()
}

func (m isActivePackageCacheMap) Get(key PackageID) (bool, bool) {
	value, ok := m.impl.Get(key)
	if !ok {
		return false, false
	}
	return value.(bool), true
}

func (m isActivePackageCacheMap) Set(key PackageID, value bool) {
	m.impl.Set(key, value, nil)
}

type parseKeysByURIMap struct {
	impl *persistent.Map
}

func newParseKeysByURIMap() parseKeysByURIMap {
	return parseKeysByURIMap{
		impl: persistent.NewMap(uriLessInterface),
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

func packageKeyLessInterface(x, y interface{}) bool {
	return packageKeyLess(x.(packageKey), y.(packageKey))
}

func packageKeyLess(x, y packageKey) bool {
	if x.mode != y.mode {
		return x.mode < y.mode
	}
	return x.id < y.id
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

// actionKeyLessInterface is the less-than relation for actionKey
// values wrapped in an interface.
func actionKeyLessInterface(a, b interface{}) bool {
	x, y := a.(actionKey), b.(actionKey)
	if x.analyzer.Name != y.analyzer.Name {
		return x.analyzer.Name < y.analyzer.Name
	}
	return x.pkgid < y.pkgid
}
