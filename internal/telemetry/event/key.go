// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

// Key is used as the identity of a Tag.
// Keys are intended to be compared by pointer only, the name should be unique
// for communicating with external systems, but it is not required or enforced.
type Key struct {
	Name        string
	Description string
}

// TagOf returns a Tag for a key and value.
// This is a trivial helper that makes common logging easier to read.
func TagOf(name string, value interface{}) Tag {
	return Tag{Key: &Key{Name: name}, Value: value}
}

// Of creates a new Tag with this key and the supplied value.
// You can use this when building a tag list.
func (k *Key) Of(v interface{}) Tag {
	return Tag{Key: k, Value: v}
}
