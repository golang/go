// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag provides support for telemetry tagging.
// This package is a thin shim over contexts with the main addition being the
// the ability to observe when contexts get tagged with new values.
package tag

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/worker"
)

//TODO: Do we need to do something more efficient than just store tags
//TODO: directly on the context?

// Tag holds a key and value pair.
// It is normally used when passing around lists of tags.
type Tag struct {
	Key   interface{}
	Value interface{}
}

// Tagger is the interface to somthing that returns a Tag given a context.
// Both Tag itself and Key support this interface, allowing methods that can
// take either (and other implementations as well)
type Tagger interface {
	// Tag returns a Tag potentially using information from the Context.
	Tag(context.Context) Tag
}

// List is a way of passing around a collection of key value pairs.
// It is an alternative to the less efficient and unordered method of using
// maps.
type List []Tag

// Observer is the type for a function that wants to be notified when new tags
// are set on a context.
// If you use context.WithValue (or equivalent) it will bypass the observers,
// you must use the setters in this package for tags that should be observed.
// Register new observers with the Observe function.
type Observer func(ctx context.Context, at time.Time, tags List)

// With is roughly equivalent to context.WithValue except that it also notifies
// registered observers.
// Unlike WithValue, it takes a list of tags so that you can set many values
// at once if needed. Each call to With results in one invocation of each
// observer.
func With(ctx context.Context, tags ...Tag) context.Context {
	at := time.Now()
	for _, t := range tags {
		ctx = context.WithValue(ctx, t.Key, t.Value)
	}
	worker.Do(func() {
		for i := len(observers) - 1; i >= 0; i-- {
			observers[i](ctx, at, tags)
		}
	})
	return ctx
}

// Get collects a set of values from the context and returns them as a tag list.
func Get(ctx context.Context, keys ...interface{}) List {
	tags := make(List, len(keys))
	for i, key := range keys {
		tags[i] = Tag{Key: key, Value: ctx.Value(key)}
	}
	return tags
}

// Tags collects a list of tags for the taggers from the context.
func Tags(ctx context.Context, taggers ...Tagger) List {
	tags := make(List, len(taggers))
	for i, t := range taggers {
		tags[i] = t.Tag(ctx)
	}
	return tags
}

var observers = []Observer{}

// Observe adds a new tag observer to the registered set.
// There is no way to ever unregister a observer.
// Observers are free to use context information to control their behavior.
func Observe(observer Observer) {
	worker.Do(func() {
		observers = append(observers, observer)
	})
}

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
func (l List) Get(k interface{}) interface{} {
	for _, t := range l {
		if t.Key == k {
			return t.Value
		}
	}
	return nil
}

// Format pretty prints a list.
// It is intended only for debugging.
func (l List) Format(f fmt.State, r rune) {
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
func (l List) Equal(other List) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) == fmt.Sprint(other)
}

// Less is intended only for using tag lists as a sorting key.
func (l List) Less(other List) bool {
	//TODO: make this more efficient
	return fmt.Sprint(l) < fmt.Sprint(other)
}
