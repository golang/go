// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"cmp"
	"iter"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
)

func finalize(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// HashString is the base hash of a string key: FNV-1a 64.
func HashString(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// HashUint64 is the base hash of an integer or bool key: splitmix64 of the
// 64-bit value (signed values use their two's-complement bits).
func HashUint64(x uint64) uint64 {
	return finalize(x + 0x9e3779b97f4a7c15)
}

// HashFloat64 is the base hash of a float key. Negative zero hashes as
// positive zero, so both spell one key (Go == treats them as equal).
// Float32 keys widen to float64 first, which is exact.
func HashFloat64(f float64) uint64 {
	if f == 0 {
		f = 0
	}
	return HashUint64(math.Float64bits(f))
}

// HashBool is the base hash of a bool key: HashUint64 of 1 or 0.
func HashBool(b bool) uint64 {
	if b {
		return HashUint64(1)
	}
	return HashUint64(0)
}

// HashCombine folds one field or element hash into a composite key hash.
func HashCombine(h, x uint64) uint64 {
	return finalize(h ^ (x + 0x9e3779b97f4a7c15 + h<<6 + h>>2))
}

// ConstMix is the bucket and slot mix: a full-diffusion avalanche of the
// base hash h under seed. Buckets use the outer salt as seed; slots use
// the bucket seed.
func ConstMix(seed, h uint64) uint64 {
	return finalize(h ^ (seed * 0x9e3779b97f4a7c15))
}

// ConstMapMaterialization is zero-initialized once-only storage for one
// top-level regular-map materialization (SPEC T-Sec7). The generated code
// declares one zero value per const map, one pure builder that rebuilds
// the full regular map, and one accessor that returns
// storage.Get(builder). The first Get runs the builder exactly once;
// later Gets reuse the same map. A builder panic repeats on every later
// Get, matching sync.OnceValue. Only sync.Once orders the single build;
// there is no per-read mutex and no package-init work.
type ConstMapMaterialization[M ~map[K]V, K comparable, V any] struct {
	once     sync.Once
	valid    bool
	panicked any
	value    M
}

// Get returns the materialized map, running build once on first use.
func (m *ConstMapMaterialization[M, K, V]) Get(build func() M) M {
	m.once.Do(func() {
		defer func() {
			m.panicked = recover()
			if !m.valid {
				panic(m.panicked)
			}
		}()
		m.value = build()
		m.valid = true
	})
	if !m.valid {
		panic(m.panicked)
	}
	return m.value
}

// ConstBucket describes one bucket of a const map table. Off and Size name
// the bucket's slot region. Seed places the bucket's keys inside it. An
// empty bucket points at the table's shared empty slot with Size 1.
type ConstBucket struct {
	Off  uint32
	Size uint32
	Seed uint64
}

// ConstSlot is one key/value slot. Occ is false for unowned slots, which a
// lookup treats as absent before it compares keys.
type ConstSlot[K comparable, V any] struct {
	Key K
	Val V
	Occ bool
}

// ConstTable is one const map's lookup table: Buckets selected by the outer
// salt, then one slot in the bucket's region.
type ConstTable[K comparable, V any] struct {
	Salt    uint64
	Buckets []ConstBucket
	Slots   []ConstSlot[K, V]
}

// Get performs the one-probe lookup for key k with base hash h: one bucket
// read, one slot probe, then full-key validation. An unoccupied slot answers
// absent before any key comparison.
func (t *ConstTable[K, V]) Get(h uint64, k K) (V, bool) {
	var zero V
	if t == nil || len(t.Buckets) == 0 {
		return zero, false
	}
	b := &t.Buckets[ConstMix(t.Salt, h)&uint64(len(t.Buckets)-1)]
	if b.Size == 0 {
		return zero, false
	}
	s := &t.Slots[b.Off+uint32(ConstMix(b.Seed, h)&uint64(b.Size-1))]
	if s.Occ && s.Key == k {
		return s.Val, true
	}
	return zero, false
}

// ConstEntry is one entry of a const map in sorted key order, the order
// range visits.
type ConstEntry[K comparable, V any] struct {
	Key K
	Val V
}

// ConstAll returns the range iterator of a const map over its sorted
// entries.
func ConstAll[K comparable, V any](sorted []ConstEntry[K, V]) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for _, e := range sorted {
			if !yield(e.Key, e.Val) {
				return
			}
		}
	}
}

// SortedAll returns the range iterator of a map over its keys in the order
// SPEC Sec 7 gives a const map: a string by its bytes, an integer or float
// by numeric value, false before true, a complex number by real part and
// then imaginary part, and a struct or array field by field or element by
// element. Keys of any other kind, and keys of different dynamic types,
// keep no defined relative order. Each captured entry is yielded once,
// without looking its key up again (a NaN key cannot be looked up).
func SortedAll[K comparable, V any](m map[K]V) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		entries := make([]ConstEntry[K, V], 0, len(m))
		var types map[reflect.Type]int
		if reflect.TypeFor[K]().Kind() == reflect.Interface {
			types = make(map[reflect.Type]int)
		}
		for k, v := range m {
			entries = append(entries, ConstEntry[K, V]{Key: k, Val: v})
			if types != nil {
				t := reflect.TypeOf(k)
				if _, ok := types[t]; !ok {
					types[t] = len(types)
				}
			}
		}
		slices.SortStableFunc(entries, func(a, b ConstEntry[K, V]) int {
			if types != nil {
				if c := cmp.Compare(types[reflect.TypeOf(a.Key)], types[reflect.TypeOf(b.Key)]); c != 0 {
					return c
				}
			}
			return compareKeys(reflect.ValueOf(a.Key), reflect.ValueOf(b.Key))
		})
		for _, entry := range entries {
			if !yield(entry.Key, entry.Val) {
				return
			}
		}
	}
}

func compareKeys(a, b reflect.Value) int {
	switch a.Kind() {
	case reflect.String:
		return strings.Compare(a.String(), b.String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return cmp.Compare(a.Int(), b.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return cmp.Compare(a.Uint(), b.Uint())
	case reflect.Float32, reflect.Float64:
		return cmp.Compare(a.Float(), b.Float())
	case reflect.Bool:
		return compareBools(a.Bool(), b.Bool())
	case reflect.Complex64, reflect.Complex128:
		x, y := a.Complex(), b.Complex()
		if c := cmp.Compare(real(x), real(y)); c != 0 {
			return c
		}
		return cmp.Compare(imag(x), imag(y))
	case reflect.Struct:
		for i := range a.NumField() {
			if c := compareKeys(a.Field(i), b.Field(i)); c != 0 {
				return c
			}
		}
	case reflect.Array:
		for i := range a.Len() {
			if c := compareKeys(a.Index(i), b.Index(i)); c != 0 {
				return c
			}
		}
	}
	return 0
}

func compareBools(a, b bool) int {
	switch {
	case a == b:
		return 0
	case b:
		return -1
	}
	return 1
}

// Opt converts a lookup result to the option form of a single-value read
// (SPEC T-202): present is some, absent is none.
func Opt[V any](v V, ok bool) Option[V] {
	return Option[V]{Value: v, Present: ok}
}

// Bare converts a lookup result to the bare option form, for payloads
// that keep nil as none (SPEC T-202): absent reads as the zero value.
func Bare[V any](v V, ok bool) V {
	if !ok {
		var zero V
		return zero
	}
	return v
}

// Materialize returns the function that builds a const map's regular map
// once, on first use, and returns that same map afterwards (SPEC T-Sec7).
// Concurrent first uses share one build.
func Materialize[M any](f func() M) func() M {
	return sync.OnceValue(f)
}
