// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package godebug makes the settings in the $GODEBUG environment variable
// available to other packages. These settings are often used for compatibility
// tweaks, when we need to change a default behavior but want to let users
// opt back in to the original. For example GODEBUG=http2server=0 disables
// HTTP/2 support in the net/http server.
//
// In typical usage, code should declare a Setting as a global
// and then call Value each time the current setting value is needed:
//
//	var http2server = godebug.New("http2server")
//
//	func ServeConn(c net.Conn) {
//		if http2server.Value() == "0" {
//			disallow HTTP/2
//			...
//		}
//		...
//	}
package godebug

import (
	"sync"
	"sync/atomic"
	_ "unsafe" // go:linkname
)

// A Setting is a single setting in the $GODEBUG environment variable.
type Setting struct {
	name  string
	once  sync.Once
	value *atomic.Pointer[string]
}

// New returns a new Setting for the $GODEBUG setting with the given name.
func New(name string) *Setting {
	return &Setting{name: name}
}

// Name returns the name of the setting.
func (s *Setting) Name() string {
	return s.name
}

// String returns a printable form for the setting: name=value.
func (s *Setting) String() string {
	return s.name + "=" + s.Value()
}

// cache is a cache of all the GODEBUG settings,
// a locked map[string]*atomic.Pointer[string].
//
// All Settings with the same name share a single
// *atomic.Pointer[string], so that when GODEBUG
// changes only that single atomic string pointer
// needs to be updated.
//
// A name appears in the values map either if it is the
// name of a Setting for which Value has been called
// at least once, or if the name has ever appeared in
// a name=value pair in the $GODEBUG environment variable.
// Once entered into the map, the name is never removed.
var cache sync.Map // name string -> value *atomic.Pointer[string]

var empty string

// Value returns the current value for the GODEBUG setting s.
//
// Value maintains an internal cache that is synchronized
// with changes to the $GODEBUG environment variable,
// making Value efficient to call as frequently as needed.
// Clients should therefore typically not attempt their own
// caching of Value's result.
func (s *Setting) Value() string {
	s.once.Do(func() {
		v, ok := cache.Load(s.name)
		if !ok {
			p := new(atomic.Pointer[string])
			p.Store(&empty)
			v, _ = cache.LoadOrStore(s.name, p)
		}
		s.value = v.(*atomic.Pointer[string])
	})
	return *s.value.Load()
}

// setUpdate is provided by package runtime.
// It calls update(def, env), where def is the default GODEBUG setting
// and env is the current value of the $GODEBUG environment variable.
// After that first call, the runtime calls update(def, env)
// again each time the environment variable changes
// (due to use of os.Setenv, for example).
//
//go:linkname setUpdate
func setUpdate(update func(string, string))

func init() {
	setUpdate(update)
}

var updateMu sync.Mutex

// update records an updated GODEBUG setting.
// def is the default GODEBUG setting for the running binary,
// and env is the current value of the $GODEBUG environment variable.
func update(def, env string) {
	updateMu.Lock()
	defer updateMu.Unlock()

	// Update all the cached values, creating new ones as needed.
	// We parse the environment variable first, so that any settings it has
	// are already locked in place (did[name] = true) before we consider
	// the defaults.
	did := make(map[string]bool)
	parse(did, env)
	parse(did, def)

	// Clear any cached values that are no longer present.
	cache.Range(func(name, v any) bool {
		if !did[name.(string)] {
			v.(*atomic.Pointer[string]).Store(&empty)
		}
		return true
	})
}

// parse parses the GODEBUG setting string s,
// which has the form k=v,k2=v2,k3=v3.
// Later settings override earlier ones.
// Parse only updates settings k=v for which did[k] = false.
// It also sets did[k] = true for settings that it updates.
func parse(did map[string]bool, s string) {
	// Scan the string backward so that later settings are used
	// and earlier settings are ignored.
	// Note that a forward scan would cause cached values
	// to temporarily use the ignored value before being
	// updated to the "correct" one.
	end := len(s)
	eq := -1
	for i := end - 1; i >= -1; i-- {
		if i == -1 || s[i] == ',' {
			if eq >= 0 {
				name, value := s[i+1:eq], s[eq+1:end]
				if !did[name] {
					did[name] = true
					v, ok := cache.Load(name)
					if !ok {
						p := new(atomic.Pointer[string])
						p.Store(&empty)
						v, _ = cache.LoadOrStore(name, p)
					}
					v.(*atomic.Pointer[string]).Store(&value)
				}
			}
			eq = -1
			end = i
		} else if s[i] == '=' {
			eq = i
		}
	}
}
