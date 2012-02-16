// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This example demonstrates a priority queue built using the heap interface.
package heap_test

import (
	"container/heap"
	"fmt"
)

// An Item is something we manage in a priority queue.
type Item struct {
	value    string // The value of the item; arbitrary.
	priority int    // The priority of the item in the queue.
	// The index is needed by changePriority and is maintained by the heap.Interface methods.
	index int // The index of the item in the heap.
}

// A PriorityQueue implements heap.Interface and holds Items.
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	// To simplify indexing expressions in these methods, we save a copy of the
	// slice object. We could instead write (*pq)[i].
	a := *pq
	n := len(a)
	a = a[0 : n+1]
	item := x.(*Item)
	item.index = n
	a[n] = item
	*pq = a
}

func (pq *PriorityQueue) Pop() interface{} {
	a := *pq
	n := len(a)
	item := a[n-1]
	item.index = -1 // for safety
	*pq = a[0 : n-1]
	return item
}

// update is not used by the example but shows how to take the top item from
// the queue, update its priority and value, and put it back.
func (pq *PriorityQueue) update(value string, priority int) {
	item := heap.Pop(pq).(*Item)
	item.value = value
	item.priority = priority
	heap.Push(pq, item)
}

// changePriority is not used by the example but shows how to change the
// priority of an arbitrary item.
func (pq *PriorityQueue) changePriority(item *Item, priority int) {
	heap.Remove(pq, item.index)
	item.priority = priority
	heap.Push(pq, item)
}

// This example pushes 10 items into a PriorityQueue and takes them out in
// order of priority.
func Example() {
	const nItem = 10
	// Random priorities for the items (a permutation of 0..9, times 11)).
	priorities := [nItem]int{
		77, 22, 44, 55, 11, 88, 33, 99, 00, 66,
	}
	values := [nItem]string{
		"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
	}
	// Create a priority queue and put some items in it.
	pq := make(PriorityQueue, 0, nItem)
	for i := 0; i < cap(pq); i++ {
		item := &Item{
			value:    values[i],
			priority: priorities[i],
		}
		heap.Push(&pq, item)
	}
	// Take the items out; should arrive in decreasing priority order.
	// For example, the highest priority (99) is the seventh item, so output starts with 99:"seven".
	for i := 0; i < nItem; i++ {
		item := heap.Pop(&pq).(*Item)
		fmt.Printf("%.2d:%s ", item.priority, item.value)
	}
	// Output:
	// 99:seven 88:five 77:zero 66:nine 55:three 44:two 33:six 22:one 11:four 00:eight
}
