// errorcheck -0 -race
// +build linux,amd64 linux,ppc64le darwin,amd64 freebsd,amd64 netbsd,amd64 windows,amd64

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

const benchmarkNumNodes = 10000

func BenchmarkUpdateNodeTransaction(b B) {
	s, nodeIDs := setupNodes(benchmarkNumNodes)
	b.ResetTimer()
	for i := 0; i < b.N(); i++ {
		_ = s.Update(func(tx1 Tx) error {
			_ = UpdateNode(tx1, &Node{
				ID: nodeIDs[i%benchmarkNumNodes],
			})
			return nil
		})
	}
}

type B interface {
	ResetTimer()
	N() int
}

type Tx interface {
}

type Node struct {
	ID string
}

type MemoryStore struct {
}

// go:noinline
func setupNodes(n int) (s *MemoryStore, nodeIDs []string) {
	return
}

//go:noinline
func (s *MemoryStore) Update(cb func(Tx) error) error {
	return nil
}

var sink interface{}

//go:noinline
func UpdateNode(tx Tx, n *Node) error {
	sink = tx
	sink = n
	return nil
}
