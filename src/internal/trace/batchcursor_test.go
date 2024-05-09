// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"strings"
	"testing"

	"slices"
)

func TestHeap(t *testing.T) {
	var heap []*batchCursor

	// Insert a bunch of values into the heap.
	checkHeap(t, heap)
	heap = heapInsert(heap, makeBatchCursor(5))
	checkHeap(t, heap)
	for i := int64(-20); i < 20; i++ {
		heap = heapInsert(heap, makeBatchCursor(i))
		checkHeap(t, heap)
	}

	// Update an element in the middle to be the new minimum.
	for i := range heap {
		if heap[i].ev.time == 5 {
			heap[i].ev.time = -21
			heapUpdate(heap, i)
			break
		}
	}
	checkHeap(t, heap)
	if heap[0].ev.time != -21 {
		t.Fatalf("heap update failed, expected %d as heap min: %s", -21, heapDebugString(heap))
	}

	// Update the minimum element to be smaller. There should be no change.
	heap[0].ev.time = -22
	heapUpdate(heap, 0)
	checkHeap(t, heap)
	if heap[0].ev.time != -22 {
		t.Fatalf("heap update failed, expected %d as heap min: %s", -22, heapDebugString(heap))
	}

	// Update the last element to be larger. There should be no change.
	heap[len(heap)-1].ev.time = 21
	heapUpdate(heap, len(heap)-1)
	checkHeap(t, heap)
	if heap[len(heap)-1].ev.time != 21 {
		t.Fatalf("heap update failed, expected %d as heap min: %s", 21, heapDebugString(heap))
	}

	// Update the last element to be smaller.
	heap[len(heap)-1].ev.time = 7
	heapUpdate(heap, len(heap)-1)
	checkHeap(t, heap)
	if heap[len(heap)-1].ev.time == 21 {
		t.Fatalf("heap update failed, unexpected %d as heap min: %s", 21, heapDebugString(heap))
	}

	// Remove an element in the middle.
	for i := range heap {
		if heap[i].ev.time == 5 {
			heap = heapRemove(heap, i)
			break
		}
	}
	checkHeap(t, heap)
	for i := range heap {
		if heap[i].ev.time == 5 {
			t.Fatalf("failed to remove heap elem with time %d: %s", 5, heapDebugString(heap))
		}
	}

	// Remove tail.
	heap = heapRemove(heap, len(heap)-1)
	checkHeap(t, heap)

	// Remove from the head, and make sure the result is sorted.
	l := len(heap)
	var removed []*batchCursor
	for i := 0; i < l; i++ {
		removed = append(removed, heap[0])
		heap = heapRemove(heap, 0)
		checkHeap(t, heap)
	}
	if !slices.IsSortedFunc(removed, (*batchCursor).compare) {
		t.Fatalf("heap elements not removed in sorted order, got: %s", heapDebugString(removed))
	}
}

func makeBatchCursor(v int64) *batchCursor {
	return &batchCursor{ev: baseEvent{time: Time(v)}}
}

func heapDebugString(heap []*batchCursor) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "[")
	for i := range heap {
		if i != 0 {
			fmt.Fprintf(&sb, ", ")
		}
		fmt.Fprintf(&sb, "%d", heap[i].ev.time)
	}
	fmt.Fprintf(&sb, "]")
	return sb.String()
}

func checkHeap(t *testing.T, heap []*batchCursor) {
	t.Helper()

	for i := range heap {
		if i == 0 {
			continue
		}
		if heap[(i-1)/2].compare(heap[i]) > 0 {
			t.Errorf("heap invariant not maintained between index %d and parent %d: %s", i, i/2, heapDebugString(heap))
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}
