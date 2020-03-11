// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

var (
	// Err is a key used to add error values to tag lists.
	Err = Key{Name: "error"}
)

// Key is used as the identity of a Tag.
// Keys are intended to be compared by pointer only, the name should be unique
// for communicating with external systems, but it is not required or enforced.
type Key struct {
	Name        string
	Description string
}

// Of creates a new Tag with this key and the supplied value.
// You can use this when building a tag list.
func (k *Key) Of(v interface{}) Tag {
	return Tag{Key: k, Value: v}
}
