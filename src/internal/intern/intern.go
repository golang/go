// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package intern lets you make smaller comparable values by boxing
// a larger comparable value (such as a 16 byte string header) down
// into a globally unique 8 byte pointer.
//
// The globally unique pointers are garbage collected with weak
// references and finalizers. This package hides that.
package intern

import (
	"internal/godebug"
	"runtime"
	"sync"
	"unsafe"
)

// A Value pointer is the handle to an underlying comparable value.
// See func Get for how Value pointers may be used.
type Value struct {
	_      [0]func() // prevent people from accidentally using value type as comparable
	cmpVal any
	// resurrected is guarded by mu (for all instances of Value).
	// It is set true whenever v is synthesized from a uintptr.
	resurrected bool
}

// Get returns the comparable value passed to the Get func
// that returned v.
func (v *Value) Get() any { return v.cmpVal }

// key is a key in our global value map.
// It contains type-specialized fields to avoid allocations
// when converting common types to empty interfaces.
type key struct {
	s      string
	cmpVal any
	// isString reports whether key contains a string.
	// Without it, the zero value of key is ambiguous.
	isString bool
}

// keyFor returns a key to use with cmpVal.
func keyFor(cmpVal any) key {
	if s, ok := cmpVal.(string); ok {
		return key{s: s, isString: true}
	}
	return key{cmpVal: cmpVal}
}

// Value returns a *Value built from k.
func (k key) Value() *Value {
	if k.isString {
		return &Value{cmpVal: k.s}
	}
	return &Value{cmpVal: k.cmpVal}
}

var (
	// mu guards valMap, a weakref map of *Value by underlying value.
	// It also guards the resurrected field of all *Values.
	mu      sync.Mutex
	valMap  = map[key]uintptr{} // to uintptr(*Value)
	valSafe = safeMap()         // non-nil in safe+leaky mode
)

var intern = godebug.New("#intern")

// safeMap returns a non-nil map if we're in safe-but-leaky mode,
// as controlled by GODEBUG=intern=leaky
func safeMap() map[key]*Value {
	if intern.Value() == "leaky" {
		return map[key]*Value{}
	}
	return nil
}

// Get returns a pointer representing the comparable value cmpVal.
//
// The returned pointer will be the same for Get(v) and Get(v2)
// if and only if v == v2, and can be used as a map key.
func Get(cmpVal any) *Value {
	return get(keyFor(cmpVal))
}

// GetByString is identical to Get, except that it is specialized for strings.
// This avoids an allocation from putting a string into an interface{}
// to pass as an argument to Get.
func GetByString(s string) *Value {
	return get(key{s: s, isString: true})
}

// We play unsafe games that violate Go's rules (and assume a non-moving
// collector). So we quiet Go here.
// See the comment below Get for more implementation details.
//
//go:nocheckptr
func get(k key) *Value {
	mu.Lock()
	defer mu.Unlock()

	var v *Value
	if valSafe != nil {
		v = valSafe[k]
	} else if addr, ok := valMap[k]; ok {
		v = (*Value)(unsafe.Pointer(addr))
		v.resurrected = true
	}
	if v != nil {
		return v
	}
	v = k.Value()
	if valSafe != nil {
		valSafe[k] = v
	} else {
		// SetFinalizer before uintptr conversion (theoretical concern;
		// see https://github.com/go4org/intern/issues/13)
		runtime.SetFinalizer(v, finalize)
		valMap[k] = uintptr(unsafe.Pointer(v))
	}
	return v
}

func finalize(v *Value) {
	mu.Lock()
	defer mu.Unlock()
	if v.resurrected {
		// We lost the race. Somebody resurrected it while we
		// were about to finalize it. Try again next round.
		v.resurrected = false
		runtime.SetFinalizer(v, finalize)
		return
	}
	delete(valMap, keyFor(v.cmpVal))
}

// Interning is simple if you don't require that unused values be
// garbage collectable. But we do require that; we don't want to be
// DOS vector. We do this by using a uintptr to hide the pointer from
// the garbage collector, and using a finalizer to eliminate the
// pointer when no other code is using it.
//
// The obvious implementation of this is to use a
// map[interface{}]uintptr-of-*interface{}, and set up a finalizer to
// delete from the map. Unfortunately, this is racy. Because pointers
// are being created in violation of Go's unsafety rules, it's
// possible to create a pointer to a value concurrently with the GC
// concluding that the value can be collected. There are other races
// that break the equality invariant as well, but the use-after-free
// will cause a runtime crash.
//
// To make this work, the finalizer needs to know that no references
// have been unsafely created since the finalizer was set up. To do
// this, values carry a "resurrected" sentinel, which gets set
// whenever a pointer is unsafely created. If the finalizer encounters
// the sentinel, it clears the sentinel and delays collection for one
// additional GC cycle, by re-installing itself as finalizer. This
// ensures that the unsafely created pointer is visible to the GC, and
// will correctly prevent collection.
//
// This technique does mean that interned values that get reused take
// at least 3 GC cycles to fully collect (1 to clear the sentinel, 1
// to clean up the unsafe map, 1 to be actually deleted).
//
// @ianlancetaylor commented in
// https://github.com/golang/go/issues/41303#issuecomment-717401656
// that it is possible to implement weak references in terms of
// finalizers without unsafe. Unfortunately, the approach he outlined
// does not work here, for two reasons. First, there is no way to
// construct a strong pointer out of a weak pointer; our map stores
// weak pointers, but we must return strong pointers to callers.
// Second, and more fundamentally, we must return not just _a_ strong
// pointer to callers, but _the same_ strong pointer to callers. In
// order to return _the same_ strong pointer to callers, we must track
// it, which is exactly what we cannot do with strong pointers.
//
// See https://github.com/inetaf/netaddr/issues/53 for more
// discussion, and https://github.com/go4org/intern/issues/2 for an
// illustration of the subtleties at play.
