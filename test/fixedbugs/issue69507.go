// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	err := run()
	if err != nil {
		panic(err)
	}
}

func run() error {
	methods := "AB"

	type node struct {
		tag     string
		choices []string
	}
	all := []node{
		{"000", permutations(methods)},
	}

	next := 1
	for len(all) > 0 {
		cur := all[0]
		k := copy(all, all[1:])
		all = all[:k]

		if len(cur.choices) == 1 {
			continue
		}

		var bestM map[byte][]string
		bMax := len(cur.choices) + 1
		bMin := -1
		for sel := range selections(methods) {
			m := make(map[byte][]string)
			for _, order := range cur.choices {
				x := findFirstMatch(order, sel)
				m[x] = append(m[x], order)
			}

			min := len(cur.choices) + 1
			max := -1
			for _, v := range m {
				if len(v) < min {
					min = len(v)
				}
				if len(v) > max {
					max = len(v)
				}
			}
			if max < bMax || (max == bMax && min > bMin) {
				bestM = m
				bMin = min
				bMax = max
			}
		}

		if bMax == len(cur.choices) {
			continue
		}

		cc := Keys(bestM)
		for c := range cc {
			choices := bestM[c]
			next++

			switch c {
			case 'A':
			case 'B':
			default:
				panic("unexpected selector type " + string(c))
			}
			all = append(all, node{"", choices})
		}
	}
	return nil
}

func permutations(s string) []string {
	if len(s) <= 1 {
		return []string{s}
	}

	var result []string
	for i, char := range s {
		rest := s[:i] + s[i+1:]
		for _, perm := range permutations(rest) {
			result = append(result, string(char)+perm)
		}
	}
	return result
}

type Seq[V any] func(yield func(V) bool)

func selections(s string) Seq[string] {
	return func(yield func(string) bool) {
		for bits := 1; bits < 1<<len(s); bits++ {
			var choice string
			for j, char := range s {
				if bits&(1<<j) != 0 {
					choice += string(char)
				}
			}
			if !yield(choice) {
				break
			}
		}
	}
}

func findFirstMatch(order, sel string) byte {
	for _, c := range order {
		return byte(c)
	}
	return 0
}

func Keys[Map ~map[K]V, K comparable, V any](m Map) Seq[K] {
	return func(yield func(K) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}
