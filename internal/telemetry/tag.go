// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"context"
	"fmt"
)

// Tag holds a key and value pair.
// It is normally used when passing around lists of tags.
type Tag struct {
	Key   interface{}
	Value interface{}
}

// TagList is a way of passing around a collection of key value pairs.
// It is an alternative to the less efficient and unordered method of using
// maps.
type TagList []Tag

// Format is used for debug printing of tags.
func (t Tag) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, `%v="%v"`, t.Key, t.Value)
}

// Get returns the tag unmodified.
// It makes Key conform to the Tagger interface.
func (t Tag) Tag(ctx context.Context) Tag {
	return t
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
