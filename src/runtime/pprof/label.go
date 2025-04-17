// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"fmt"
	"slices"
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
		return labelMap{}
	}
	return *labels
}

// labelMap is the representation of the label set held in the context type.
// This is an initial implementation, but it will be replaced with something
// that admits incremental immutable modification more efficiently.
type labelMap struct {
	LabelSet
}

// String satisfies Stringer and returns key, value pairs in a consistent
// order.
func (l *labelMap) String() string {
	if l == nil {
		return ""
	}
	keyVals := make([]string, 0, len(l.list))

	for _, lbl := range l.list {
		keyVals = append(keyVals, fmt.Sprintf("%q:%q", lbl.key, lbl.value))
	}

	slices.Sort(keyVals)
	return "{" + strings.Join(keyVals, ", ") + "}"
}

// WithLabels returns a new [context.Context] with the given labels added.
// A label overwrites a prior label with the same key.
func WithLabels(ctx context.Context, labels LabelSet) context.Context {
	parentLabels := labelValue(ctx)
	return context.WithValue(ctx, labelContextKey{}, &labelMap{mergeLabelSets(parentLabels.LabelSet, labels)})
}

func mergeLabelSets(left, right LabelSet) LabelSet {
	if len(left.list) == 0 {
		return right
	} else if len(right.list) == 0 {
		return left
	}

	l, r := 0, 0
	result := make([]label, 0, len(right.list))
	for l < len(left.list) && r < len(right.list) {
		switch strings.Compare(left.list[l].key, right.list[r].key) {
		case -1: // left key < right key
			result = append(result, left.list[l])
			l++
		case 1: // right key < left key
			result = append(result, right.list[r])
			r++
		case 0: // keys are equal, right value overwrites left value
			result = append(result, right.list[r])
			l++
			r++
		}
	}

	// Append the remaining elements
	result = append(result, left.list[l:]...)
	result = append(result, right.list[r:]...)

	return LabelSet{list: result}
}

// Labels takes an even number of strings representing key-value pairs
// and makes a [LabelSet] containing them.
// A label overwrites a prior label with the same key.
// Currently only the CPU and goroutine profiles utilize any labels
// information.
// See https://golang.org/issue/23458 for details.
func Labels(args ...string) LabelSet {
	if len(args)%2 != 0 {
		panic("uneven number of arguments to pprof.Labels")
	}
	list := make([]label, 0, len(args)/2)
	sortedNoDupes := true
	for i := 0; i+1 < len(args); i += 2 {
		list = append(list, label{key: args[i], value: args[i+1]})
		sortedNoDupes = sortedNoDupes && (i < 2 || args[i] > args[i-2])
	}
	if !sortedNoDupes {
		// slow path: keys are unsorted, contain duplicates, or both
		slices.SortStableFunc(list, func(a, b label) int {
			return strings.Compare(a.key, b.key)
		})
		deduped := make([]label, 0, len(list))
		for i, lbl := range list {
			if i == 0 || lbl.key != list[i-1].key {
				deduped = append(deduped, lbl)
			} else {
				deduped[len(deduped)-1] = lbl
			}
		}
		list = deduped
	}
	return LabelSet{list: list}
}

// Label returns the value of the label with the given key on ctx, and a boolean indicating
// whether that label exists.
func Label(ctx context.Context, key string) (string, bool) {
	ctxLabels := labelValue(ctx)
	for _, lbl := range ctxLabels.list {
		if lbl.key == key {
			return lbl.value, true
		}
	}
	return "", false
}

// ForLabels invokes f with each label set on the context.
// The function f should return true to continue iteration or false to stop iteration early.
func ForLabels(ctx context.Context, f func(key, value string) bool) {
	ctxLabels := labelValue(ctx)
	for _, lbl := range ctxLabels.list {
		if !f(lbl.key, lbl.value) {
			break
		}
	}
}
