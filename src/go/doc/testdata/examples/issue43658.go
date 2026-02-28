// Copyright ©2016 The Gonum Authors. All rights reserved.
// Copyright 2021 The Go Authors. All rights reserved.
// (above line required for our license-header checker)
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package community_test

import (
	"fmt"
	"log"
	"sort"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/internal/ordered"
	"gonum.org/v1/gonum/graph/simple"
)

func ExampleProfile_simple() {
	// Profile calls Modularize which implements the Louvain modularization algorithm.
	// Since this is a randomized algorithm we use a defined random source to ensure
	// consistency between test runs. In practice, results will not differ greatly
	// between runs with different PRNG seeds.
	src := rand.NewSource(1)

	// Create dumbell graph:
	//
	//  0       4
	//  |\     /|
	//  | 2 - 3 |
	//  |/     \|
	//  1       5
	//
	g := simple.NewUndirectedGraph()
	for u, e := range smallDumbell {
		for v := range e {
			g.SetEdge(simple.Edge{F: simple.Node(u), T: simple.Node(v)})
		}
	}

	// Get the profile of internal node weight for resolutions
	// between 0.1 and 10 using logarithmic bisection.
	p, err := community.Profile(
		community.ModularScore(g, community.Weight, 10, src),
		true, 1e-3, 0.1, 10,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Print out each step with communities ordered.
	for _, d := range p {
		comm := d.Communities()
		for _, c := range comm {
			sort.Sort(ordered.ByID(c))
		}
		sort.Sort(ordered.BySliceIDs(comm))
		fmt.Printf("Low:%.2v High:%.2v Score:%v Communities:%v Q=%.3v\n",
			d.Low, d.High, d.Score, comm, community.Q(g, comm, d.Low))
	}

	// Output:
	// Low:0.1 High:0.29 Score:14 Communities:[[0 1 2 3 4 5]] Q=0.9
	// Low:0.29 High:2.3 Score:12 Communities:[[0 1 2] [3 4 5]] Q=0.714
	// Low:2.3 High:3.5 Score:4 Communities:[[0 1] [2] [3] [4 5]] Q=-0.31
	// Low:3.5 High:10 Score:0 Communities:[[0] [1] [2] [3] [4] [5]] Q=-0.607
}

// intset is an integer set.
type intset map[int]struct{}

func linksTo(i ...int) intset {
	if len(i) == 0 {
		return nil
	}
	s := make(intset)
	for _, v := range i {
		s[v] = struct{}{}
	}
	return s
}

var (
	smallDumbell = []intset{
		0: linksTo(1, 2),
		1: linksTo(2),
		2: linksTo(3),
		3: linksTo(4, 5),
		4: linksTo(5),
		5: nil,
	}

	// http://www.slate.com/blogs/the_world_/2014/07/17/the_middle_east_friendship_chart.html
	middleEast = struct{ friends, complicated, enemies []intset }{
		// green cells
		friends: []intset{
			0:  nil,
			1:  linksTo(5, 7, 9, 12),
			2:  linksTo(11),
			3:  linksTo(4, 5, 10),
			4:  linksTo(3, 5, 10),
			5:  linksTo(1, 3, 4, 8, 10, 12),
			6:  nil,
			7:  linksTo(1, 12),
			8:  linksTo(5, 9, 11),
			9:  linksTo(1, 8, 12),
			10: linksTo(3, 4, 5),
			11: linksTo(2, 8),
			12: linksTo(1, 5, 7, 9),
		},

		// yellow cells
		complicated: []intset{
			0:  linksTo(2, 4),
			1:  linksTo(4, 8),
			2:  linksTo(0, 3, 4, 5, 8, 9),
			3:  linksTo(2, 8, 11),
			4:  linksTo(0, 1, 2, 8),
			5:  linksTo(2),
			6:  nil,
			7:  linksTo(9, 11),
			8:  linksTo(1, 2, 3, 4, 10, 12),
			9:  linksTo(2, 7, 11),
			10: linksTo(8),
			11: linksTo(3, 7, 9, 12),
			12: linksTo(8, 11),
		},

		// red cells
		enemies: []intset{
			0:  linksTo(1, 3, 5, 6, 7, 8, 9, 10, 11, 12),
			1:  linksTo(0, 2, 3, 6, 10, 11),
			2:  linksTo(1, 6, 7, 10, 12),
			3:  linksTo(0, 1, 6, 7, 9, 12),
			4:  linksTo(6, 7, 9, 11, 12),
			5:  linksTo(0, 6, 7, 9, 11),
			6:  linksTo(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
			7:  linksTo(0, 2, 3, 4, 5, 6, 8, 10),
			8:  linksTo(0, 6, 7),
			9:  linksTo(0, 3, 4, 5, 6, 10),
			10: linksTo(0, 1, 2, 6, 7, 9, 11, 12),
			11: linksTo(0, 1, 4, 5, 6, 10),
			12: linksTo(0, 2, 3, 4, 6, 10),
		},
	}
)

var friends, enemies *simple.WeightedUndirectedGraph

func init() {
	friends = simple.NewWeightedUndirectedGraph(0, 0)
	for u, e := range middleEast.friends {
		// Ensure unconnected nodes are included.
		if friends.Node(int64(u)) == nil {
			friends.AddNode(simple.Node(u))
		}
		for v := range e {
			friends.SetWeightedEdge(simple.WeightedEdge{F: simple.Node(u), T: simple.Node(v), W: 1})
		}
	}
	enemies = simple.NewWeightedUndirectedGraph(0, 0)
	for u, e := range middleEast.enemies {
		// Ensure unconnected nodes are included.
		if enemies.Node(int64(u)) == nil {
			enemies.AddNode(simple.Node(u))
		}
		for v := range e {
			enemies.SetWeightedEdge(simple.WeightedEdge{F: simple.Node(u), T: simple.Node(v), W: -1})
		}
	}
}

func ExampleProfile_multiplex() {
	// Profile calls ModularizeMultiplex which implements the Louvain modularization
	// algorithm. Since this is a randomized algorithm we use a defined random source
	// to ensure consistency between test runs. In practice, results will not differ
	// greatly between runs with different PRNG seeds.
	src := rand.NewSource(1)

	// The undirected graphs, friends and enemies, are the political relationships
	// in the Middle East as described in the Slate article:
	// http://www.slate.com/blogs/the_world_/2014/07/17/the_middle_east_friendship_chart.html
	g, err := community.NewUndirectedLayers(friends, enemies)
	if err != nil {
		log.Fatal(err)
	}
	weights := []float64{1, -1}

	// Get the profile of internal node weight for resolutions
	// between 0.1 and 10 using logarithmic bisection.
	p, err := community.Profile(
		community.ModularMultiplexScore(g, weights, true, community.WeightMultiplex, 10, src),
		true, 1e-3, 0.1, 10,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Print out each step with communities ordered.
	for _, d := range p {
		comm := d.Communities()
		for _, c := range comm {
			sort.Sort(ordered.ByID(c))
		}
		sort.Sort(ordered.BySliceIDs(comm))
		fmt.Printf("Low:%.2v High:%.2v Score:%v Communities:%v Q=%.3v\n",
			d.Low, d.High, d.Score, comm, community.QMultiplex(g, comm, weights, []float64{d.Low}))
	}

	// Output:
	// Low:0.1 High:0.72 Score:26 Communities:[[0] [1 7 9 12] [2 8 11] [3 4 5 10] [6]] Q=[24.7 1.97]
	// Low:0.72 High:1.1 Score:24 Communities:[[0 6] [1 7 9 12] [2 8 11] [3 4 5 10]] Q=[16.9 14.1]
	// Low:1.1 High:1.2 Score:18 Communities:[[0 2 6 11] [1 7 9 12] [3 4 5 8 10]] Q=[9.16 25.1]
	// Low:1.2 High:1.6 Score:10 Communities:[[0 3 4 5 6 10] [1 7 9 12] [2 8 11]] Q=[10.5 26.7]
	// Low:1.6 High:1.6 Score:8 Communities:[[0 1 6 7 9 12] [2 8 11] [3 4 5 10]] Q=[5.56 39.8]
	// Low:1.6 High:1.8 Score:2 Communities:[[0 2 3 4 5 6 10] [1 7 8 9 11 12]] Q=[-1.82 48.6]
	// Low:1.8 High:2.3 Score:-6 Communities:[[0 2 3 4 5 6 8 10 11] [1 7 9 12]] Q=[-5 57.5]
	// Low:2.3 High:2.4 Score:-10 Communities:[[0 1 2 6 7 8 9 11 12] [3 4 5 10]] Q=[-11.2 79]
	// Low:2.4 High:4.3 Score:-52 Communities:[[0 1 2 3 4 5 6 7 8 9 10 11 12]] Q=[-46.1 117]
	// Low:4.3 High:10 Score:-54 Communities:[[0 1 2 3 4 6 7 8 9 10 11 12] [5]] Q=[-82 254]
}
