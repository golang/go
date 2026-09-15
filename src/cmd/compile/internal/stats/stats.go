// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stats provides a convenient way to collect compiler statistics.
package stats

import (
	"fmt"
	"log"
	"sort"
	"strings"
)

type Stats struct {
	stats []*PrefixStats
}

func (s *Stats) Merge(other *Stats) {
	if s == nil || other == nil {
		return
	}
	allKeys := map[string]struct{}{}
	for _, s := range s.stats {
		allKeys[s.prefix] = struct{}{}
	}
	for _, s := range other.stats {
		allKeys[s.prefix] = struct{}{}
	}
	newStats := []*PrefixStats{}
	for k := range allKeys {
		newS := &PrefixStats{prefix: k, stats: make(map[string]int64)}
		for _, s1 := range s.stats {
			if s1.prefix == k {
				newS.Merge(s1)
			}
		}
		for _, s2 := range other.stats {
			if s2.prefix == k {
				newS.Merge(s2)
			}
		}
		newStats = append(newStats, newS)
	}
	s.stats = newStats
}

func (s *Stats) Print() {
	if s == nil {
		return
	}
	for _, s := range s.stats {
		s.Print()
	}
}

func (s *Stats) NewPrefixStat(p string) *PrefixStats {
	if s == nil {
		return nil
	}
	pstat := &PrefixStats{prefix: p, stats: make(map[string]int64)}
	s.stats = append(s.stats, pstat)
	return pstat
}

// Stats holds a collection of statistics.
type PrefixStats struct {
	prefix string
	stats  map[string]int64
}

// Record records a value for a given statistic.
func (s *PrefixStats) Record(name string, value int64) {
	if s == nil {
		return
	}
	s.stats[name] += value
}

// Merge merges another Stats object into this one.
func (s *PrefixStats) Merge(other *PrefixStats) {
	if s == nil || other == nil {
		return
	}
	for name, value := range other.stats {
		s.stats[name] += value
	}
}

// Print prints the collected statistics to the console.
func (s *PrefixStats) Print() {
	if s == nil {
		return
	}
	var names []string
	for name := range s.stats {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	fmt.Fprintln(&b, "Compiler statistics:")
	for _, name := range names {
		fmt.Fprintf(&b, "%s-%-30s: %d\n", s.prefix, name, s.stats[name])
	}
	log.Print(b.String())
}
