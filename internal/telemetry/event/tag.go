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

// TagList is a way of passing around a collection of key value pairs.
// It is an alternative to the less efficient and unordered method of using
// maps.
type TagList []Tag

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

// FindAll returns corresponding tags for each key in keys.
// If no tag is found for a key, the Tag at its corresponding
// index will be of the zero value.
func (l TagList) FindAll(keys []Key) TagList {
	tags := make(TagList, len(keys))
	for i, key := range keys {
		tags[i] = l.find(key.Identity())
	}
	return tags
}

func (l TagList) find(key interface{}) Tag {
	for _, t := range l {
		if t.key == key {
			return t
		}
	}
	return Tag{}
}

// Format pretty prints a list.
// It is intended only for debugging.
func (l TagList) Format(f fmt.State, r rune) {
	printed := false
	for _, t := range l {
		if t.value == nil {
			continue
		}
		if printed {
			fmt.Fprint(f, ",")
		}
		fmt.Fprint(f, t)
		printed = true
	}
}

// Equal returns true if two lists are identical.
func (l TagList) Equal(other TagList) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) == fmt.Sprint(other)
}

// Less is intended only for using tag lists as a sorting key.
func (l TagList) Less(other TagList) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) < fmt.Sprint(other)
}
