// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"reflect"
	"sort"
	"testing"
)

func labelsSorted(ctx context.Context) []label {
	ls := []label{}
	ForLabels(ctx, func { key, value ->
		ls = append(ls, label{key, value})
		return true
	})
	sort.Sort(labelSorter(ls))
	return ls
}

type labelSorter []label

func (s labelSorter) Len() int           { return len(s) }
func (s labelSorter) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s labelSorter) Less(i, j int) bool { return s[i].key < s[j].key }

func TestContextLabels(t *testing.T) {
	// Background context starts with no labels.
	ctx := context.Background()
	labels := labelsSorted(ctx)
	if len(labels) != 0 {
		t.Errorf("labels on background context: want [], got %v ", labels)
	}

	// Add a single label.
	ctx = WithLabels(ctx, Labels("key", "value"))
	// Retrieve it with Label.
	v, ok := Label(ctx, "key")
	if !ok || v != "value" {
		t.Errorf(`Label(ctx, "key"): got %v, %v; want "value", ok`, v, ok)
	}
	gotLabels := labelsSorted(ctx)
	wantLabels := []label{{"key", "value"}}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("(sorted) labels on context: got %v, want %v", gotLabels, wantLabels)
	}

	// Add a label with a different key.
	ctx = WithLabels(ctx, Labels("key2", "value2"))
	v, ok = Label(ctx, "key2")
	if !ok || v != "value2" {
		t.Errorf(`Label(ctx, "key2"): got %v, %v; want "value2", ok`, v, ok)
	}
	gotLabels = labelsSorted(ctx)
	wantLabels = []label{{"key", "value"}, {"key2", "value2"}}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("(sorted) labels on context: got %v, want %v", gotLabels, wantLabels)
	}

	// Add label with first key to test label replacement.
	ctx = WithLabels(ctx, Labels("key", "value3"))
	v, ok = Label(ctx, "key")
	if !ok || v != "value3" {
		t.Errorf(`Label(ctx, "key3"): got %v, %v; want "value3", ok`, v, ok)
	}
	gotLabels = labelsSorted(ctx)
	wantLabels = []label{{"key", "value3"}, {"key2", "value2"}}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("(sorted) labels on context: got %v, want %v", gotLabels, wantLabels)
	}

	// Labels called with two labels with the same key should pick the second.
	ctx = WithLabels(ctx, Labels("key4", "value4a", "key4", "value4b"))
	v, ok = Label(ctx, "key4")
	if !ok || v != "value4b" {
		t.Errorf(`Label(ctx, "key4"): got %v, %v; want "value4b", ok`, v, ok)
	}
	gotLabels = labelsSorted(ctx)
	wantLabels = []label{{"key", "value3"}, {"key2", "value2"}, {"key4", "value4b"}}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("(sorted) labels on context: got %v, want %v", gotLabels, wantLabels)
	}
}

func TestLabelMapStringer(t *testing.T) {
	for _, tbl := range []struct {
		m        labelMap
		expected string
	}{
		{
			m: labelMap{
				// empty map
			},
			expected: "{}",
		}, {
			m: labelMap{
				"foo": "bar",
			},
			expected: `{"foo":"bar"}`,
		}, {
			m: labelMap{
				"foo":             "bar",
				"key1":            "value1",
				"key2":            "value2",
				"key3":            "value3",
				"key4WithNewline": "\nvalue4",
			},
			expected: `{"foo":"bar", "key1":"value1", "key2":"value2", "key3":"value3", "key4WithNewline":"\nvalue4"}`,
		},
	} {
		if got := tbl.m.String(); tbl.expected != got {
			t.Errorf("%#v.String() = %q; want %q", tbl.m, got, tbl.expected)
		}
	}
}
