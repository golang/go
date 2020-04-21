// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"fmt"
	"io"
	"reflect"
	"unsafe"
)

// Tag holds a key and value pair.
// It is normally used when passing around lists of tags.
type Tag struct {
	key     Key
	packed  uint64
	untyped interface{}
}

// TagMap is the interface to a collection of Tags indexed by key.
type TagMap interface {
	// Find returns the tag that matches the supplied key.
	Find(key Key) Tag
}

// TagList is the interface to something that provides an iterable
// list of tags.
// Iteration should start from 0 and continue until Valid returns false.
type TagList interface {
	// Valid returns true if the index is within range for the list.
	// It does not imply the tag at that index will itself be valid.
	Valid(index int) bool
	// Tag returns the tag at the given index.
	Tag(index int) Tag
}

// tagList implements TagList for a list of Tags.
type tagList struct {
	tags []Tag
}

// tagFilter wraps a TagList filtering out specific tags.
type tagFilter struct {
	keys       []Key
	underlying TagList
}

// tagMap implements TagMap for a simple list of tags.
type tagMap struct {
	tags []Tag
}

// tagMapChain implements TagMap for a list of underlying TagMap.
type tagMapChain struct {
	maps []TagMap
}

// TagOfValue creates a new tag from the key and value.
// This method is for implementing new key types, tag creation should
// normally be done with the Of method of the key.
func TagOfValue(k Key, value interface{}) Tag { return Tag{key: k, untyped: value} }

// UnpackValue assumes the tag was built using TagOfValue and returns the value
// that was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Tag) UnpackValue() interface{} { return t.untyped }

// TagOf64 creates a new tag from a key and a uint64. This is often
// used for non uint64 values that can be packed into a uint64.
// This method is for implementing new key types, tag creation should
// normally be done with the Of method of the key.
func TagOf64(k Key, v uint64) Tag { return Tag{key: k, packed: v} }

// Unpack64 assumes the tag was built using TagOf64 and returns the value that
// was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Tag) Unpack64() uint64 { return t.packed }

// TagOfString creates a new tag from a key and a string.
// This method is for implementing new key types, tag creation should
// normally be done with the Of method of the key.
func TagOfString(k Key, v string) Tag {
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&v))
	return Tag{
		key:     k,
		packed:  uint64(hdr.Len),
		untyped: unsafe.Pointer(hdr.Data),
	}
}

// UnpackString assumes the tag was built using TagOfString and returns the
// value that was passed to that constructor.
// This method is for implementing new key types, for type safety normal
// access should be done with the From method of the key.
func (t Tag) UnpackString() string {
	var v string
	hdr := (*reflect.StringHeader)(unsafe.Pointer(&v))
	hdr.Data = uintptr(t.untyped.(unsafe.Pointer))
	hdr.Len = int(t.packed)
	return *(*string)(unsafe.Pointer(hdr))
}

// Valid returns true if the Tag is a valid one (it has a key).
func (t Tag) Valid() bool { return t.key != nil }

// Key returns the key of this Tag.
func (t Tag) Key() Key { return t.key }

// Format is used for debug printing of tags.
func (t Tag) Format(f fmt.State, r rune) {
	if !t.Valid() {
		io.WriteString(f, `nil`)
		return
	}
	io.WriteString(f, t.Key().Name())
	io.WriteString(f, "=")
	var buf [128]byte
	t.Key().Format(f, buf[:0], t)
}

func (l *tagList) Valid(index int) bool {
	return index >= 0 && index < len(l.tags)
}

func (l *tagList) Tag(index int) Tag {
	return l.tags[index]
}

func (f *tagFilter) Valid(index int) bool {
	return f.underlying.Valid(index)
}

func (f *tagFilter) Tag(index int) Tag {
	tag := f.underlying.Tag(index)
	for _, f := range f.keys {
		if tag.Key() == f {
			return Tag{}
		}
	}
	return tag
}

func (l tagMap) Find(key Key) Tag {
	for _, tag := range l.tags {
		if tag.Key() == key {
			return tag
		}
	}
	return Tag{}
}

func (c tagMapChain) Find(key Key) Tag {
	for _, src := range c.maps {
		tag := src.Find(key)
		if tag.Valid() {
			return tag
		}
	}
	return Tag{}
}

var emptyList = &tagList{}

func NewTagList(tags ...Tag) TagList {
	if len(tags) == 0 {
		return emptyList
	}
	return &tagList{tags: tags}
}

func Filter(l TagList, keys ...Key) TagList {
	if len(keys) == 0 {
		return l
	}
	return &tagFilter{keys: keys, underlying: l}
}

func NewTagMap(tags ...Tag) TagMap {
	return tagMap{tags: tags}
}

func MergeTagMaps(srcs ...TagMap) TagMap {
	var nonNil []TagMap
	for _, src := range srcs {
		if src != nil {
			nonNil = append(nonNil, src)
		}
	}
	if len(nonNil) == 1 {
		return nonNil[0]
	}
	return tagMapChain{maps: nonNil}
}
