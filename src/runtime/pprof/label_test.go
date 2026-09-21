// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"fmt"
	"internal/runtime/pprof/label"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func labelsSorted(ctx context.Context) []label.Label {
	ls := []label.Label{}
	ForLabels(ctx, func(key, value string) bool {
		ls = append(ls, label.Label{Key: key, Value: value})
		return true
	})
	slices.SortFunc(ls, func(a, b label.Label) int { return strings.Compare(a.Key, b.Key) })
	return ls
}

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
	wantLabels := []label.Label{{Key: "key", Value: "value"}}
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
	wantLabels = []label.Label{{Key: "key", Value: "value"}, {Key: "key2", Value: "value2"}}
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
	wantLabels = []label.Label{{Key: "key", Value: "value3"}, {Key: "key2", Value: "value2"}}
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
	wantLabels = []label.Label{{Key: "key", Value: "value3"}, {Key: "key2", Value: "value2"}, {Key: "key4", Value: "value4b"}}
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
				label.NewSet(Labels("foo", "bar").list),
			},
			expected: `{"foo":"bar"}`,
		}, {
			m: labelMap{
				label.NewSet(Labels(
					"foo", "bar",
					"key1", "value1",
					"key2", "value2",
					"key3", "value3",
					"key4WithNewline", "\nvalue4",
				).list),
			},
			expected: `{"foo":"bar", "key1":"value1", "key2":"value2", "key3":"value3", "key4WithNewline":"\nvalue4"}`,
		},
	} {
		if got := tbl.m.String(); tbl.expected != got {
			t.Errorf("%#v.String() = %q; want %q", tbl.m, got, tbl.expected)
		}
	}
}

func BenchmarkLabels(b *testing.B) {
	b.Run("set-one", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			Do(context.Background(), Labels("key", "value"), func(context.Context) {})
		}
	})

	b.Run("merge-one", func(b *testing.B) {
		ctx := WithLabels(context.Background(), Labels("key1", "val1"))

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			Do(ctx, Labels("key2", "value2"), func(context.Context) {})
		}
	})

	b.Run("overwrite-one", func(b *testing.B) {
		ctx := WithLabels(context.Background(), Labels("key", "val"))

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			Do(ctx, Labels("key", "value"), func(context.Context) {})
		}
	})

	for _, scenario := range []string{"ordered", "unordered"} {
		var labels []string
		for i := 0; i < 10; i++ {
			labels = append(labels, fmt.Sprintf("key%03d", i), fmt.Sprintf("value%03d", i))
		}
		if scenario == "unordered" {
			labels[0], labels[len(labels)-1] = labels[len(labels)-1], labels[0]
		}

		b.Run(scenario, func(b *testing.B) {
			b.Run("set-many", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Do(context.Background(), Labels(labels...), func(context.Context) {})
				}
			})

			b.Run("merge-many", func(b *testing.B) {
				ctx := WithLabels(context.Background(), Labels(labels[:len(labels)/2]...))

				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					Do(ctx, Labels(labels[len(labels)/2:]...), func(context.Context) {})
				}
			})

			b.Run("overwrite-many", func(b *testing.B) {
				ctx := WithLabels(context.Background(), Labels(labels...))

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					Do(ctx, Labels(labels...), func(context.Context) {})
				}
			})
		})
	}

	// TODO: hit slow path in Labels
}
