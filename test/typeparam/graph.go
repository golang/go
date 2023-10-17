// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
)

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _SliceEqual[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// A Graph is a collection of nodes. A node may have an arbitrary number
// of edges. An edge connects two nodes. Both nodes and edges must be
// comparable. This is an undirected simple graph.
type _Graph[_Node _NodeC[_Edge], _Edge _EdgeC[_Node]] struct {
	nodes []_Node
}

// _NodeC is the constraints on a node in a graph, given the _Edge type.
type _NodeC[_Edge any] interface {
	comparable
	Edges() []_Edge
}

// Edgec is the constraints on an edge in a graph, given the _Node type.
type _EdgeC[_Node any] interface {
	comparable
	Nodes() (a, b _Node)
}

// _New creates a new _Graph from a collection of Nodes.
func _New[_Node _NodeC[_Edge], _Edge _EdgeC[_Node]](nodes []_Node) *_Graph[_Node, _Edge] {
	return &_Graph[_Node, _Edge]{nodes: nodes}
}

// nodePath holds the path to a node during ShortestPath.
// This should ideally be a type defined inside ShortestPath,
// but the translator tool doesn't support that.
type nodePath[_Node _NodeC[_Edge], _Edge _EdgeC[_Node]] struct {
	node _Node
	path []_Edge
}

// ShortestPath returns the shortest path between two nodes,
// as an ordered list of edges. If there are multiple shortest paths,
// which one is returned is unpredictable.
func (g *_Graph[_Node, _Edge]) ShortestPath(from, to _Node) ([]_Edge, error) {
	visited := make(map[_Node]bool)
	visited[from] = true
	workqueue := []nodePath[_Node, _Edge]{nodePath[_Node, _Edge]{from, nil}}
	for len(workqueue) > 0 {
		current := workqueue
		workqueue = nil
		for _, np := range current {
			edges := np.node.Edges()
			for _, edge := range edges {
				a, b := edge.Nodes()
				if a == np.node {
					a = b
				}
				if !visited[a] {
					ve := append([]_Edge(nil), np.path...)
					ve = append(ve, edge)
					if a == to {
						return ve, nil
					}
					workqueue = append(workqueue, nodePath[_Node, _Edge]{a, ve})
					visited[a] = true
				}
			}
		}
	}
	return nil, errors.New("no path")
}

type direction int

const (
	north direction = iota
	ne
	east
	se
	south
	sw
	west
	nw
	up
	down
)

func (dir direction) String() string {
	strs := map[direction]string{
		north: "north",
		ne:    "ne",
		east:  "east",
		se:    "se",
		south: "south",
		sw:    "sw",
		west:  "west",
		nw:    "nw",
		up:    "up",
		down:  "down",
	}
	if str, ok := strs[dir]; ok {
		return str
	}
	return fmt.Sprintf("direction %d", dir)
}

type mazeRoom struct {
	index int
	exits [10]int
}

type mazeEdge struct {
	from, to int
	dir      direction
}

// Edges returns the exits from the room.
func (m mazeRoom) Edges() []mazeEdge {
	var r []mazeEdge
	for i, exit := range m.exits {
		if exit != 0 {
			r = append(r, mazeEdge{
				from: m.index,
				to:   exit,
				dir:  direction(i),
			})
		}
	}
	return r
}

// Nodes returns the rooms connected by an edge.
//go:noinline
func (e mazeEdge) Nodes() (mazeRoom, mazeRoom) {
	m1, ok := zork[e.from]
	if !ok {
		panic("bad edge")
	}
	m2, ok := zork[e.to]
	if !ok {
		panic("bad edge")
	}
	return m1, m2
}

// The first maze in Zork. Room indexes based on original Fortran data file.
// You are in a maze of twisty little passages, all alike.
var zork = map[int]mazeRoom{
	11: {exits: [10]int{north: 11, south: 12, east: 14}}, // west to Troll Room
	12: {exits: [10]int{south: 11, north: 14, east: 13}},
	13: {exits: [10]int{west: 12, north: 14, up: 16}},
	14: {exits: [10]int{west: 13, north: 11, east: 15}},
	15: {exits: [10]int{south: 14}},                   // Dead End
	16: {exits: [10]int{east: 17, north: 13, sw: 18}}, // skeleton, etc.
	17: {exits: [10]int{west: 16}},                    // Dead End
	18: {exits: [10]int{down: 16, east: 19, west: 18, up: 22}},
	19: {exits: [10]int{up: 29, west: 18, ne: 15, east: 20, south: 30}},
	20: {exits: [10]int{ne: 19, west: 20, se: 21}},
	21: {exits: [10]int{north: 20}}, // Dead End
	22: {exits: [10]int{north: 18, east: 24, down: 23, south: 28, west: 26, nw: 22}},
	23: {exits: [10]int{east: 22, west: 28, up: 24}},
	24: {exits: [10]int{ne: 25, down: 23, nw: 28, sw: 26}},
	25: {exits: [10]int{sw: 24}}, // Grating room (up to Clearing)
	26: {exits: [10]int{west: 16, sw: 24, east: 28, up: 22, north: 27}},
	27: {exits: [10]int{south: 26}}, // Dead End
	28: {exits: [10]int{east: 22, down: 26, south: 23, west: 24}},
	29: {exits: [10]int{west: 30, nw: 29, ne: 19, south: 19}},
	30: {exits: [10]int{west: 29, south: 19}}, // ne to Cyclops Room
}

func TestShortestPath() {
	// The Zork maze is not a proper undirected simple graph,
	// as there are some one way paths (e.g., 19 -> 15),
	// but for this test that doesn't matter.

	// Set the index field in the map. Simpler than doing it in the
	// composite literal.
	for k := range zork {
		r := zork[k]
		r.index = k
		zork[k] = r
	}

	var nodes []mazeRoom
	for idx, room := range zork {
		mridx := room
		mridx.index = idx
		nodes = append(nodes, mridx)
	}
	g := _New[mazeRoom, mazeEdge](nodes)
	path, err := g.ShortestPath(zork[11], zork[30])
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	var steps []direction
	for _, edge := range path {
		steps = append(steps, edge.dir)
	}
	want := []direction{east, west, up, sw, east, south}
	if !_SliceEqual(steps, want) {
		panic(fmt.Sprintf("ShortestPath returned %v, want %v", steps, want))
	}
}

func main() {
	TestShortestPath()
}
