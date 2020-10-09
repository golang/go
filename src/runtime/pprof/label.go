// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type label struct {
	key   string
	value string
}

// LabelSet is a set of labels.
type LabelSet struct {
	list []label
}

// labelContextKey is the type of contextKeys used for profiler labels.
type labelContextKey struct{}

func labelValue(ctx context.Context) labelMap {
	labels, _ := ctx.Value(labelContextKey{}).(*labelMap)
	if labels == nil {
		return labelMap(nil)
	}
	return *labels
}

// labelMap is the representation of the label set held in the context type.
// This is an initial implementation, but it will be replaced with something
// that admits incremental immutable modification more efficiently.
type labelMap map[string]string

// String statisfies Stringer and returns key, value pairs in a consistent
// order.
func (l *labelMap) String() string {
	if l == nil {
		return ""
	}
	keyVals := make([]string, 0, len(*l))

	for k, v := range *l {
		keyVals = append(keyVals, fmt.Sprintf("%q:%q", k, v))
	}

	sort.Strings(keyVals)

	return "{" + strings.Join(keyVals, ", ") + "}"
}

// WithLabels returns a new context.Context with the given labels added.
// A label overwrites a prior label with the same key.
func WithLabels(ctx context.Context, labels LabelSet) context.Context {
	childLabels := make(labelMap)
	parentLabels := labelValue(ctx)
	// TODO(matloob): replace the map implementation with something
	// more efficient so creating a child context WithLabels doesn't need
	// to clone the map.
	for k, v := range parentLabels {
		childLabels[k] = v
	}
	for _, label := range labels.list {
		childLabels[label.key] = label.value
	}
	return context.WithValue(ctx, labelContextKey{}, &childLabels)
}

// Labels takes an even number of strings representing key-value pairs
// and makes a LabelSet containing them.
// A label overwrites a prior label with the same key.
// Currently only the CPU and goroutine profiles utilize any labels
// information.
// See https://golang.org/issue/23458 for details.
func Labels(args ...string) LabelSet {
	if len(args)%2 != 0 {
		panic("uneven number of arguments to pprof.Labels")
	}
	list := make([]label, 0, len(args)/2)
	for i := 0; i+1 < len(args); i += 2 {
		list = append(list, label{key: args[i], value: args[i+1]})
	}
	return LabelSet{list: list}
}

// Label returns the value of the label with the given key on ctx, and a boolean indicating
// whether that label exists.
func Label(ctx context.Context, key string) (string, bool) {
	ctxLabels := labelValue(ctx)
	v, ok := ctxLabels[key]
	return v, ok
}

// ForLabels invokes f with each label set on the context.
// The function f should return true to continue iteration or false to stop iteration early.
func ForLabels(ctx context.Context, f func(key, value string) bool) {
	ctxLabels := labelValue(ctx)
	for k, v := range ctxLabels {
		if !f(k, v) {
			break
		}
	}
}
