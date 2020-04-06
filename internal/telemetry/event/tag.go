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

// TagPointer is the interface to something that provides an iterable
// list of tags.
type TagPointer interface {
	// Next advances to the next entry in the list and return a TagIterator for it.
	// It will return nil if there are no more entries.
	Next() TagPointer
	// Tag returns the tag the pointer is for.
	Tag() Tag
}

// TagIterator is used to iterate through tags using TagPointer.
// It is a small helper that will normally fully inline to make it easier to
// manage the fact that pointer advance returns a new pointer rather than
// moving the existing one.
type TagIterator struct {
	ptr TagPointer
}

// tagPointer implements TagPointer over a simple list of tags.
type tagPointer struct {
	tags []Tag
}

// tagPointer wraps a TagPointer filtering out specific tags.
type tagFilter struct {
	filter     []Key
	underlying TagPointer
}

// tagPointerChain implements TagMap for a list of underlying TagMap.
type tagPointerChain struct {
	ptrs []TagPointer
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

func (i *TagIterator) Valid() bool {
	return i.ptr != nil
}

func (i *TagIterator) Advance() {
	i.ptr = i.ptr.Next()
}

func (i *TagIterator) Tag() Tag {
	return i.ptr.Tag()
}

func (i tagPointer) Next() TagPointer {
	// loop until we are on a valid tag
	for {
		// move on one tag
		i.tags = i.tags[1:]
		// check if we have exhausted the current list
		if len(i.tags) == 0 {
			// no more tags, so no more iterator
			return nil
		}
		// if the tag is valid, we are done
		if i.tags[0].Valid() {
			return i
		}
	}
}

func (i tagPointer) Tag() Tag {
	return i.tags[0]
}

func (i tagFilter) Next() TagPointer {
	// loop until we are on a valid tag
	for {
		i.underlying = i.underlying.Next()
		if i.underlying == nil {
			return nil
		}
		if !i.filtered() {
			return i
		}
	}
}

func (i tagFilter) filtered() bool {
	tag := i.underlying.Tag()
	for _, f := range i.filter {
		if tag.Key == f {
			return true
		}
	}
	return false
}

func (i tagFilter) Tag() Tag {
	return i.underlying.Tag()
}

func (i tagPointerChain) Next() TagPointer {
	i.ptrs[0] = i.ptrs[0].Next()
	if i.ptrs[0] == nil {
		i.ptrs = i.ptrs[1:]
	}
	if len(i.ptrs) == 0 {
		return nil
	}
	return i
}

func (i tagPointerChain) Tag() Tag {
	return i.ptrs[0].Tag()
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

func NewTagIterator(tags ...Tag) TagIterator {
	if len(tags) == 0 {
		return TagIterator{}
	}
	result := TagIterator{ptr: tagPointer{tags: tags}}
	if !result.Tag().Valid() {
		result.Advance()
	}
	return result
}

func Filter(it TagIterator, keys ...Key) TagIterator {
	if !it.Valid() || len(keys) == 0 {
		return it
	}
	ptr := tagFilter{filter: keys, underlying: it.ptr}
	result := TagIterator{ptr: ptr}
	if ptr.filtered() {
		result.Advance()
	}
	return result
}

func ChainTagIterators(iterators ...TagIterator) TagIterator {
	if len(iterators) == 0 {
		return TagIterator{}
	}
	ptrs := make([]TagPointer, 0, len(iterators))
	for _, it := range iterators {
		if it.Valid() {
			ptrs = append(ptrs, it.ptr)
		}
	}
	if len(ptrs) == 0 {
		return TagIterator{}
	}
	return TagIterator{ptr: tagPointerChain{ptrs: ptrs}}
}

func NewTagMap(tags ...Tag) TagMap {
	return tagMap{tags: tags}
}

func MergeTagMaps(srcs ...TagMap) TagMap {
	return tagMapChain{maps: srcs}
}
