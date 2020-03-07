// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

import (
	"context"
)

// Key represents the key for a context tag.
// It is a helper to make use of context tagging slightly easier to read, it is
// not strictly needed to use it at all.
// It is intended that your common tagging keys are declared as constants of
// this type, and then you can use the methods of this type to apply and find
// those values in the context.
type Key string

// TagOf returns a Tag for a key and value.
// This is a trivial helper that makes common logging easier to read.
func TagOf(key interface{}, value interface{}) Tag {
	return Tag{Key: key, Value: value}
}

// Of creates a new Tag with this key and the supplied value.
// You can use this when building a tag list.
func (k Key) Of(v interface{}) Tag {
	return Tag{Key: k, Value: v}
}

// From can be used to get a tag for the key from a context.
func (k Key) From(ctx context.Context) Tag {
	return Tag{Key: k, Value: ctx.Value(k)}
}

// With is a wrapper over the Label package level function for just this key.
func (k Key) With(ctx context.Context, v interface{}) context.Context {
	return Label(ctx, Tag{Key: k, Value: v})
}
