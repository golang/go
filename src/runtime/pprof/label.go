// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"fmt"
	"internal/runtime/pprof/label"
	"slices"
	"strings"
)

// LabelSet is a set of labels.
type LabelSet struct {
	list []label.Label
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
	label.Set
}

// String satisfies Stringer and returns key, value pairs in a consistent
// order.
func (l *labelMap) String() string {
	if l == nil {
		return ""
	}
	keyVals := make([]string, 0, len(l.Set.List))

	for _, lbl := range l.Set.List {
		keyVals = append(keyVals, fmt.Sprintf("%q:%q", lbl.Key, lbl.Value))
	}

	slices.Sort(keyVals)
	return "{" + strings.Join(keyVals, ", ") + "}"
}

// WithLabels returns a new [context.Context] with the given labels added.
// A label overwrites a prior label with the same key.
func WithLabels(ctx context.Context, labels LabelSet) context.Context {
	parentLabels := labelValue(ctx)
	return context.WithValue(ctx, labelContextKey{}, &labelMap{mergeLabelSets(parentLabels.Set, labels)})
}

func mergeLabelSets(left label.Set, right LabelSet) label.Set {
	if len(left.List) == 0 {
		return label.NewSet(right.list)
	} else if len(right.list) == 0 {
		return left
	}

	lList, rList := left.List, right.list
	l, r := 0, 0
	result := make([]label.Label, 0, len(rList))
	for l < len(lList) && r < len(rList) {
		switch strings.Compare(lList[l].Key, rList[r].Key) {
		case -1: // left key < right key
			result = append(result, lList[l])
			l++
		case 1: // right key < left key
			result = append(result, rList[r])
			r++
		case 0: // keys are equal, right value overwrites left value
			result = append(result, rList[r])
			l++
			r++
		}
	}

	// Append the remaining elements
	result = append(result, lList[l:]...)
	result = append(result, rList[r:]...)

	return label.NewSet(result)
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
	list := make([]label.Label, 0, len(args)/2)
	sortedNoDupes := true
	for i := 0; i+1 < len(args); i += 2 {
		list = append(list, label.Label{Key: args[i], Value: args[i+1]})
		sortedNoDupes = sortedNoDupes && (i < 2 || args[i] > args[i-2])
	}
	if !sortedNoDupes {
		// slow path: keys are unsorted, contain duplicates, or both
		slices.SortStableFunc(list, func(a, b label.Label) int {
			return strings.Compare(a.Key, b.Key)
		})
		deduped := make([]label.Label, 0, len(list))
		for i, lbl := range list {
			if i == 0 || lbl.Key != list[i-1].Key {
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
	for _, lbl := range ctxLabels.Set.List {
		if lbl.Key == key {
			return lbl.Value, true
		}
	}
	return "", false
}

// ForLabels invokes f with each label set on the context.
// The function f should return true to continue iteration or false to stop iteration early.
func ForLabels(ctx context.Context, f func(key, value string) bool) {
	ctxLabels := labelValue(ctx)
	for _, lbl := range ctxLabels.Set.List {
		if !f(lbl.Key, lbl.Value) {
			break
		}
	}
}
