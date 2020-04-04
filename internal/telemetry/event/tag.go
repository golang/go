// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"fmt"
)

// Tag holds a key and value pair.
// It is normally used when passing around lists of tags.
type Tag struct {
	Key     Key
	packed  uint64
	str     string
	untyped interface{}
}

// TagMap is the interface to a collection of Tags indexed by key.
type TagMap interface {
	// Find returns the tag that matches the supplied key.
	Find(key interface{}) Tag
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

// Valid returns true if the Tag is a valid one (it has a key).
func (t Tag) Valid() bool { return t.Key != nil }

// Format is used for debug printing of tags.
func (t Tag) Format(f fmt.State, r rune) {
	if !t.Valid() {
		fmt.Fprintf(f, `nil`)
		return
	}
	switch key := t.Key.(type) {
	case *IntKey:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *Int8Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *Int16Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *Int32Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *Int64Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *UIntKey:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *UInt8Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *UInt16Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *UInt32Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *UInt64Key:
		fmt.Fprintf(f, "%s=%d", key.Name(), key.From(t))
	case *Float32Key:
		fmt.Fprintf(f, "%s=%g", key.Name(), key.From(t))
	case *Float64Key:
		fmt.Fprintf(f, "%s=%g", key.Name(), key.From(t))
	case *BooleanKey:
		fmt.Fprintf(f, "%s=%t", key.Name(), key.From(t))
	case *StringKey:
		fmt.Fprintf(f, "%s=%q", key.Name(), key.From(t))
	case *ErrorKey:
		fmt.Fprintf(f, "%s=%v", key.Name(), key.From(t))
	case *ValueKey:
		fmt.Fprintf(f, "%s=%v", key.Name(), key.From(t))
	default:
		fmt.Fprintf(f, `%s="invalid type %T"`, key.Name(), key)
	}
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
		if tag.Key == f {
			return Tag{}
		}
	}
	return tag
}

func (l tagMap) Find(key interface{}) Tag {
	for _, tag := range l.tags {
		if tag.Key == key {
			return tag
		}
	}
	return Tag{}
}

func (c tagMapChain) Find(key interface{}) Tag {
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
	return tagMapChain{maps: srcs}
}
