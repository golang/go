// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math/rand"
	. "runtime"
	"testing"
	"unsafe"
)

type MyNode struct {
	LFNode
	data int
}

// allocMyNode allocates nodes that are stored in an lfstack
// outside the Go heap.
// We require lfstack objects to live outside the heap so that
// checkptr passes on the unsafe shenanigans used.
func allocMyNode(data int) *MyNode {
	n := (*MyNode)(PersistentAlloc(unsafe.Sizeof(MyNode{}), TagAlign))
	LFNodeValidate(&n.LFNode)
	n.data = data
	return n
}

func fromMyNode(node *MyNode) *LFNode {
	return (*LFNode)(unsafe.Pointer(node))
}

func toMyNode(node *LFNode) *MyNode {
	return (*MyNode)(unsafe.Pointer(node))
}

var global any

func TestLFStack(t *testing.T) {
	stack := new(uint64)
	global = stack // force heap allocation

	// Check the stack is initially empty.
	if LFStackPop(stack) != nil {
		t.Fatalf("stack is not empty")
	}

	// Push one element.
	node := allocMyNode(42)
	LFStackPush(stack, fromMyNode(node))

	// Push another.
	node = allocMyNode(43)
	LFStackPush(stack, fromMyNode(node))

	// Pop one element.
	node = toMyNode(LFStackPop(stack))
	if node == nil {
		t.Fatalf("stack is empty")
	}
	if node.data != 43 {
		t.Fatalf("no lifo")
	}

	// Pop another.
	node = toMyNode(LFStackPop(stack))
	if node == nil {
		t.Fatalf("stack is empty")
	}
	if node.data != 42 {
		t.Fatalf("no lifo")
	}

	// Check the stack is empty again.
	if LFStackPop(stack) != nil {
		t.Fatalf("stack is not empty")
	}
	if *stack != 0 {
		t.Fatalf("stack is not empty")
	}
}

func TestLFStackStress(t *testing.T) {
	const K = 100
	P := 4 * GOMAXPROCS(-1)
	N := 100000
	if testing.Short() {
		N /= 10
	}
	// Create 2 stacks.
	stacks := [2]*uint64{new(uint64), new(uint64)}
	// Push K elements randomly onto the stacks.
	sum := 0
	for i := 0; i < K; i++ {
		sum += i
		node := allocMyNode(i)
		LFStackPush(stacks[i%2], fromMyNode(node))
	}
	c := make(chan bool, P)
	for p := 0; p < P; p++ {
		go func() {
			r := rand.New(rand.NewSource(rand.Int63()))
			// Pop a node from a random stack, then push it onto a random stack.
			for i := 0; i < N; i++ {
				node := toMyNode(LFStackPop(stacks[r.Intn(2)]))
				if node != nil {
					LFStackPush(stacks[r.Intn(2)], fromMyNode(node))
				}
			}
			c <- true
		}()
	}
	for i := 0; i < P; i++ {
		<-c
	}
	// Pop all elements from both stacks, and verify that nothing lost.
	sum2 := 0
	cnt := 0
	for i := 0; i < 2; i++ {
		for {
			node := toMyNode(LFStackPop(stacks[i]))
			if node == nil {
				break
			}
			cnt++
			sum2 += node.data
			node.Next = 0
		}
	}
	if cnt != K {
		t.Fatalf("Wrong number of nodes %d/%d", cnt, K)
	}
	if sum2 != sum {
		t.Fatalf("Wrong sum %d/%d", sum2, sum)
	}
}
