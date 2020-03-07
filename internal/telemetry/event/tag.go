// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
	"fmt"
)

// Tag holds a key and value pair.
// It is normally used when passing around lists of tags.
type Tag struct {
	Key   *Key
	Value interface{}
}

// TagList is a way of passing around a collection of key value pairs.
// It is an alternative to the less efficient and unordered method of using
// maps.
type TagList []Tag

// Format is used for debug printing of tags.
func (t Tag) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, `%v="%v"`, t.Key.Name, t.Value)
}

// Tags collects a set of values from the context and returns them as a tag list.
func Tags(ctx context.Context, keys ...*Key) TagList {
	tags := make(TagList, len(keys))
	for i, key := range keys {
		tags[i] = Tag{Key: key, Value: ctx.Value(key)}
	}
	return tags
}

// Get will get a single key's value from the list.
func (l TagList) Get(k interface{}) interface{} {
	for _, t := range l {
		if t.Key == k {
			return t.Value
		}
	}
	return nil
}

// Format pretty prints a list.
// It is intended only for debugging.
func (l TagList) Format(f fmt.State, r rune) {
	printed := false
	for _, t := range l {
		if t.Value == nil {
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
