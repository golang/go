// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"sync"
	"sync/atomic"
)

// This file contains reference map implementations for unit-tests.

// mapInterface is the interface Map implements.
type mapInterface interface {
	Load(any) (any, bool)
	Store(key, value any)
	LoadOrStore(key, value any) (actual any, loaded bool)
	LoadAndDelete(key any) (value any, loaded bool)
	Delete(any)
	Range(func(key, value any) (shouldContinue bool))
}

// RWMutexMap is an implementation of mapInterface using a sync.RWMutex.
type RWMutexMap struct {
	mu    sync.RWMutex
	dirty map[any]any
}

func (m *RWMutexMap) Load(key any) (value any, ok bool) {
	m.mu.RLock()
	value, ok = m.dirty[key]
	m.mu.RUnlock()
	return
}

func (m *RWMutexMap) Store(key, value any) {
	m.mu.Lock()
	if m.dirty == nil {
		m.dirty = make(map[any]any)
	}
	m.dirty[key] = value
	m.mu.Unlock()
}

func (m *RWMutexMap) LoadOrStore(key, value any) (actual any, loaded bool) {
	m.mu.Lock()
	actual, loaded = m.dirty[key]
	if !loaded {
		actual = value
		if m.dirty == nil {
			m.dirty = make(map[any]any)
		}
		m.dirty[key] = value
	}
	m.mu.Unlock()
	return actual, loaded
}

func (m *RWMutexMap) LoadAndDelete(key any) (value any, loaded bool) {
	m.mu.Lock()
	value, loaded = m.dirty[key]
	if !loaded {
		m.mu.Unlock()
		return nil, false
	}
	delete(m.dirty, key)
	m.mu.Unlock()
	return value, loaded
}

func (m *RWMutexMap) Delete(key any) {
	m.mu.Lock()
	delete(m.dirty, key)
	m.mu.Unlock()
}

func (m *RWMutexMap) Range(f func(key, value any) (shouldContinue bool)) {
	m.mu.RLock()
	keys := make([]any, 0, len(m.dirty))
	for k := range m.dirty {
		keys = append(keys, k)
	}
	m.mu.RUnlock()

	for _, k := range keys {
		v, ok := m.Load(k)
		if !ok {
			continue
		}
		if !f(k, v) {
			break
		}
	}
}

// DeepCopyMap is an implementation of mapInterface using a Mutex and
// atomic.Value.  It makes deep copies of the map on every write to avoid
// acquiring the Mutex in Load.
type DeepCopyMap struct {
	mu    sync.Mutex
	clean atomic.Value
}

func (m *DeepCopyMap) Load(key any) (value any, ok bool) {
	clean, _ := m.clean.Load().(map[any]any)
	value, ok = clean[key]
	return value, ok
}

func (m *DeepCopyMap) Store(key, value any) {
	m.mu.Lock()
	dirty := m.dirty()
	dirty[key] = value
	m.clean.Store(dirty)
	m.mu.Unlock()
}

func (m *DeepCopyMap) LoadOrStore(key, value any) (actual any, loaded bool) {
	clean, _ := m.clean.Load().(map[any]any)
	actual, loaded = clean[key]
	if loaded {
		return actual, loaded
	}

	m.mu.Lock()
	// Reload clean in case it changed while we were waiting on m.mu.
	clean, _ = m.clean.Load().(map[any]any)
	actual, loaded = clean[key]
	if !loaded {
		dirty := m.dirty()
		dirty[key] = value
		actual = value
		m.clean.Store(dirty)
	}
	m.mu.Unlock()
	return actual, loaded
}

func (m *DeepCopyMap) LoadAndDelete(key any) (value any, loaded bool) {
	m.mu.Lock()
	dirty := m.dirty()
	value, loaded = dirty[key]
	delete(dirty, key)
	m.clean.Store(dirty)
	m.mu.Unlock()
	return
}

func (m *DeepCopyMap) Delete(key any) {
	m.mu.Lock()
	dirty := m.dirty()
	delete(dirty, key)
	m.clean.Store(dirty)
	m.mu.Unlock()
}

func (m *DeepCopyMap) Range(f func(key, value any) (shouldContinue bool)) {
	clean, _ := m.clean.Load().(map[any]any)
	for k, v := range clean {
		if !f(k, v) {
			break
		}
	}
}

func (m *DeepCopyMap) dirty() map[any]any {
	clean, _ := m.clean.Load().(map[any]any)
	dirty := make(map[any]any, len(clean)+1)
	for k, v := range clean {
		dirty[k] = v
	}
	return dirty
}
