// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package label

import (
	"fmt"
	"io"
	"reflect"
	"unsafe"
)

// Key is used as the identity of a Label.
// Keys are intended to be compared by pointer only, the name should be unique
// for communicating with external systems, but it is not required or enforced.
type Key interface {
	// Name returns the key name.
	Name() string
	// Description returns a string that can be used to describe the value.
	Description() string

	// Format is used in formatting to append the value of the label to the
	// supplied buffer.
	// The formatter may use the supplied buf as a scratch area to avoid
	// allocations.
	Format(w io.Writer, buf []byte, l Label)
}

// Label holds a key and value pair.
// It is normally used when passing around lists of labels.
type Label struct {
	key     Key
	packed  uint64
	untyped interface{}
}

// Map is the interface to a collection of Labels indexed by key.
type Map interface {
	// Find returns the label that matches the supplied key.
	Find(key Key) Label
}

// List is the interface to something that provides an iterable
// list of labels.
// Iteration should start from 0 and continue until Valid returns false.
type List interface {
	// Valid returns true if the index is within range for the list.
	// It does not imply the label at that index will itself be valid.
	Valid(index int) bool
	// Label returns the label at the given index.
	Label(index int) Label
}

// list implements LabelList for a list of Labels.
type list struct {
	labels []Label
}

// filter wraps a LabelList filtering out specific labels.
type filter struct {
	keys       []Key
	underlying List
}

// listMap implements LabelMap for a simple list of labels.
type listMap struct {
	labels []Label
}

// mapChain implements LabelMap for a list of underlying LabelMap.
type mapChain struct {
	maps []Map
}

// OfValue creates a new label from the key and value.
// This method is for implementing new key types, label creation should
// normally be done with the Of method of the key.
func OfValue(k Key, value interface{}) Label { return Label{key: k, untyped: value} }

// UnpackValue assumes the label was built using LabelOfValue and returns the value
// that was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Label) UnpackValue() interface{} { return t.untyped }

// Of64 creates a new label from a key and a uint64. This is often
// used for non uint64 values that can be packed into a uint64.
// This method is for implementing new key types, label creation should
// normally be done with the Of method of the key.
func Of64(k Key, v uint64) Label { return Label{key: k, packed: v} }

// Unpack64 assumes the label was built using LabelOf64 and returns the value that
// was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Label) Unpack64() uint64 { return t.packed }

// OfString creates a new label from a key and a string.
// This method is for implementing new key types, label creation should
// normally be done with the Of method of the key.
func OfString(k Key, v string) Label {
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&v))
	return Label{
		key:     k,
		packed:  uint64(hdr.Len),
		untyped: unsafe.Pointer(hdr.Data),
	}
}

// UnpackString assumes the label was built using LabelOfString and returns the
// value that was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Label) UnpackString() string {
	var v string
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&v))
	hdr.Data = uintptr(t.untyped.(unsafe.Pointer))
	hdr.Len = int(t.packed)
	return *(*string)(unsafe.Pointer(hdr))
}

// Valid returns true if the Label is a valid one (it has a key).
func (t Label) Valid() bool { return t.key != nil }

// Key returns the key of this Label.
func (t Label) Key() Key { return t.key }

// Format is used for debug printing of labels.
func (t Label) Format(f fmt.State, r rune) {
	if !t.Valid() {
		io.WriteString(f, `nil`)
		return
	}
	io.WriteString(f, t.Key().Name())
	io.WriteString(f, "=")
	var buf [128]byte
	t.Key().Format(f, buf[:0], t)
}

func (l *list) Valid(index int) bool {
	return index >= 0 && index < len(l.labels)
}

func (l *list) Label(index int) Label {
	return l.labels[index]
}

func (f *filter) Valid(index int) bool {
	return f.underlying.Valid(index)
}

func (f *filter) Label(index int) Label {
	l := f.underlying.Label(index)
	for _, f := range f.keys {
		if l.Key() == f {
			return Label{}
		}
	}
	return l
}

func (lm listMap) Find(key Key) Label {
	for _, l := range lm.labels {
		if l.Key() == key {
			return l
		}
	}
	return Label{}
}

func (c mapChain) Find(key Key) Label {
	for _, src := range c.maps {
		l := src.Find(key)
		if l.Valid() {
			return l
		}
	}
	return Label{}
}

var emptyList = &list{}

func NewList(labels ...Label) List {
	if len(labels) == 0 {
		return emptyList
	}
	return &list{labels: labels}
}

func Filter(l List, keys ...Key) List {
	if len(keys) == 0 {
		return l
	}
	return &filter{keys: keys, underlying: l}
}

func NewMap(labels ...Label) Map {
	return listMap{labels: labels}
}

func MergeMaps(srcs ...Map) Map {
	var nonNil []Map
	for _, src := range srcs {
		if src != nil {
			nonNil = append(nonNil, src)
		}
	}
	if len(nonNil) == 1 {
		return nonNil[0]
	}
	return mapChain{maps: nonNil}
}
