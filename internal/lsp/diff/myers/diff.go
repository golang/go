// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package myers implements the Myers diff algorithm.
package myers

import "strings"

// Sources:
// https://blog.jcoglan.com/2017/02/17/the-myers-diff-algorithm-part-3/
// https://www.codeproject.com/Articles/42279/%2FArticles%2F42279%2FInvestigating-Myers-diff-algorithm-Part-1-of-2

type Op struct {
	Kind    OpKind
	Content []string // content from b
	I1, I2  int      // indices of the line in a
	J1      int      // indices of the line in b, J2 implied by len(Content)
}

type OpKind int

const (
	Delete OpKind = iota
	Insert
	Equal
)

func (k OpKind) String() string {
	switch k {
	case Delete:
		return "delete"
	case Insert:
		return "insert"
	case Equal:
		return "equal"
	default:
		panic("unknown operation kind")
	}
}

func ApplyEdits(a []string, operations []*Op) []string {
	var b []string
	var prevI2 int
	for _, op := range operations {
		// catch up to latest indices
		if op.I1-prevI2 > 0 {
			for _, c := range a[prevI2:op.I1] {
				b = append(b, c)
			}
		}
		switch op.Kind {
		case Equal, Insert:
			b = append(b, op.Content...)
		}
		prevI2 = op.I2
	}
	// final catch up
	if len(a)-prevI2 > 0 {
		for _, c := range a[prevI2:len(a)] {
			b = append(b, c)
		}
	}
	return b
}

// Operations returns the list of operations to convert a into b, consolidating
// operations for multiple lines and not including equal lines.
func Operations(a, b []string) []*Op {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}

	trace, offset := shortestEditSequence(a, b)
	snakes := backtrack(trace, len(a), len(b), offset)

	M, N := len(a), len(b)

	var i int
	solution := make([]*Op, len(a)+len(b))

	add := func(op *Op, i2, j2 int) {
		if op == nil {
			return
		}
		op.I2 = i2
		if op.Kind == Insert {
			op.Content = b[op.J1:j2]
		}
		solution[i] = op
		i++
	}
	x, y := 0, 0
	for _, snake := range snakes {
		if len(snake) < 2 {
			continue
		}
		var op *Op
		// delete (horizontal)
		for snake[0]-snake[1] > x-y {
			if op == nil {
				op = &Op{
					Kind: Delete,
					I1:   x,
					J1:   y,
				}
			}
			x++
			if x == M {
				break
			}
		}
		add(op, x, y)
		op = nil
		// insert (vertical)
		for snake[0]-snake[1] < x-y {
			if op == nil {
				op = &Op{
					Kind: Insert,
					I1:   x,
					J1:   y,
				}
			}
			y++
		}
		add(op, x, y)
		op = nil
		// equal (diagonal)
		for x < snake[0] {
			x++
			y++
		}
		if x >= M && y >= N {
			break
		}
	}
	return solution[:i]
}

// backtrack uses the trace for the edit sequence computation and returns the
// "snakes" that make up the solution. A "snake" is a single deletion or
// insertion followed by zero or diagonals.
func backtrack(trace [][]int, x, y, offset int) [][]int {
	snakes := make([][]int, len(trace))
	d := len(trace) - 1
	for ; x > 0 && y > 0 && d > 0; d-- {
		V := trace[d]
		if len(V) == 0 {
			continue
		}
		snakes[d] = []int{x, y}

		k := x - y

		var kPrev int
		if k == -d || (k != d && V[k-1+offset] < V[k+1+offset]) {
			kPrev = k + 1
		} else {
			kPrev = k - 1
		}

		x = V[kPrev+offset]
		y = x - kPrev
	}
	if x < 0 || y < 0 {
		return snakes
	}
	snakes[d] = []int{x, y}
	return snakes
}

// shortestEditSequence returns the shortest edit sequence that converts a into b.
func shortestEditSequence(a, b []string) ([][]int, int) {
	M, N := len(a), len(b)
	V := make([]int, 2*(N+M)+1)
	offset := N + M
	trace := make([][]int, N+M+1)

	// Iterate through the maximum possible length of the SES (N+M).
	for d := 0; d <= N+M; d++ {
		copyV := make([]int, len(V))
		// k lines are represented by the equation y = x - k. We move in
		// increments of 2 because end points for even d are on even k lines.
		for k := -d; k <= d; k += 2 {
			// At each point, we either go down or to the right. We go down if
			// k == -d, and we go to the right if k == d. We also prioritize
			// the maximum x value, because we prefer deletions to insertions.
			var x int
			if k == -d || (k != d && V[k-1+offset] < V[k+1+offset]) {
				x = V[k+1+offset] // down
			} else {
				x = V[k-1+offset] + 1 // right
			}

			y := x - k

			// Diagonal moves while we have equal contents.
			for x < M && y < N && a[x] == b[y] {
				x++
				y++
			}

			V[k+offset] = x

			// Return if we've exceeded the maximum values.
			if x == M && y == N {
				// Makes sure to save the state of the array before returning.
				copy(copyV, V)
				trace[d] = copyV
				return trace, offset
			}
		}

		// Save the state of the array.
		copy(copyV, V)
		trace[d] = copyV
	}
	return nil, 0
}

func SplitLines(text string) []string {
	lines := strings.SplitAfter(text, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}
