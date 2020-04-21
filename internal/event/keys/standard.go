// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package keys

var (
	// Msg is a key used to add message strings to label lists.
	Msg = NewString("message", "a readable message")
	// Name is used for things like traces that have a name.
	Name = NewString("name", "an entity name")
	// Err is a key used to add error values to label lists.
	Err = NewError("error", "an error that occurred")
)
