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
	key   *key
	value interface{}
}

// TagSet is a collection of Tags.
// It provides a way to create new tag sets by adding new tags to an existing
// set, and preserves the order in which tags were added when iterating.
// Tags can also be searched for in the set by their key.
type TagSet struct {
	list tagList
}

type tagList struct {
	tags []Tag
	next *tagList
}

// TagIterator is used to iterate through all the tags in a TagSet.
type TagIterator struct {
	list  tagList
	index int
}

// Key returns the key for this Tag.
func (t Tag) Key() Key { return t.key }

// Value returns the value for this Tag.
func (t Tag) Value() interface{} { return t.value }

// Format is used for debug printing of tags.
func (t Tag) Format(f fmt.State, r rune) {
	if t.key == nil {
		fmt.Fprintf(f, `nil`)
		return
	}
	fmt.Fprintf(f, `%v="%v"`, t.key.name, t.value)
}

func newTagSet(tags []Tag) TagSet {
	return TagSet{list: tagList{tags: tags}}
}

// FindAll returns corresponding tags for each key in keys.
// The resulting TagSet will have one entry for each key in the same order
// as they were passed in, and if no tag is found for a key Tag at its
// corresponding index will also have no value.
func (s TagSet) FindAll(keys []Key) TagSet {
	tags := make([]Tag, len(keys))
	for i, key := range keys {
		tags[i] = s.find(key.Identity())
	}
	return TagSet{list: tagList{tags: tags}}
}

func (s TagSet) find(key interface{}) Tag {
	//TODO: do we want/need a faster access pattern?
	for i := s.Iterator(); i.Next(); {
		tag := i.Value()
		if tag.key == key {
			return tag
		}
	}
	return Tag{}
}

// Format pretty prints a list.
// It is intended only for debugging.
func (s TagSet) Format(f fmt.State, r rune) {
	printed := false
	for i := s.Iterator(); i.Next(); {
		tag := i.Value()
		if tag.value == nil {
			continue
		}
		if printed {
			fmt.Fprint(f, ",")
		}
		fmt.Fprint(f, tag)
		printed = true
	}
}

// Add returns a new TagSet where the supplied tags are included and
// override any tags already in this TagSet.
func (s TagSet) Add(tags ...Tag) TagSet {
	if len(tags) <= 0 {
		// we don't allow empty tag lists in the chain
		return s
	}
	if len(s.list.tags) <= 0 {
		// adding to an empty list, no need for a chain
		s.list.tags = tags
		return s
	}
	// we need to add a new entry to the head of the list
	old := s.list
	s.list.next = &old
	s.list.tags = tags
	return s
}

// IsEmpty returns true if the TagSet contains no tags.
func (s TagSet) IsEmpty() bool {
	// the only way the head can be empty is if there is no chain
	return len(s.list.tags) <= 0
}

// Iterator returns an iterator for this TagSet.
func (s TagSet) Iterator() TagIterator {
	return TagIterator{list: s.list, index: -1}
}

// Next advances the iterator onto the next tag.
// It returns true if the iterator is still valid.
func (i *TagIterator) Next() bool {
	// advance the iterator
	i.index++
	if i.index < len(i.list.tags) {
		// within range of the tags, so next was valid
		return true
	}
	if i.list.next == nil {
		// no more lists in the chain, iterator no longer valid
		return false
	}
	// need to move on to the next list in the chain
	i.list = *i.list.next
	i.index = 0
	return true
}

// Value returns the tag the iterator is currently pointing to.
// It is an error to call this on an iterator that is not valid.
// You must have called Next and checked the return value before
// calling this method.
func (i *TagIterator) Value() Tag {
	return i.list.tags[i.index]
}

// Set can be used to replace the tag currently being pointed to.
func (i TagIterator) Set(tag Tag) {
	i.list.tags[i.index] = tag
}

// Equal returns true if two lists are identical.
func (l TagSet) Equal(other TagSet) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) == fmt.Sprint(other)
}

// Less is intended only for using tag lists as a sorting key.
func (l TagSet) Less(other TagSet) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) < fmt.Sprint(other)
}
