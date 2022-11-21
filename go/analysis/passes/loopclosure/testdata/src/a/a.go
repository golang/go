// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the loopclosure checker.

package testdata

import (
	"sync"

	"golang.org/x/sync/errgroup"
)

var A int

func _() {
	var s []int
	for i, v := range s {
		go func() {
			println(i) // want "loop variable i captured by func literal"
			println(v) // want "loop variable v captured by func literal"
		}()
	}
	for i, v := range s {
		defer func() {
			println(i) // want "loop variable i captured by func literal"
			println(v) // want "loop variable v captured by func literal"
		}()
	}
	for i := range s {
		go func() {
			println(i) // want "loop variable i captured by func literal"
		}()
	}
	for _, v := range s {
		go func() {
			println(v) // want "loop variable v captured by func literal"
		}()
	}
	for i, v := range s {
		go func() {
			println(i, v)
		}()
		println("unfortunately, we don't catch the error above because of this statement")
	}
	for i, v := range s {
		go func(i, v int) {
			println(i, v)
		}(i, v)
	}
	for i, v := range s {
		i, v := i, v
		go func() {
			println(i, v)
		}()
	}

	// iteration variable declared outside the loop
	for A = range s {
		go func() {
			println(A) // want "loop variable A captured by func literal"
		}()
	}
	// iteration variable declared in a different file
	for B = range s {
		go func() {
			println(B) // want "loop variable B captured by func literal"
		}()
	}
	// If the key of the range statement is not an identifier
	// the code should not panic (it used to).
	var x [2]int
	var f int
	for x[0], f = range s {
		go func() {
			_ = f // want "loop variable f captured by func literal"
		}()
	}
	type T struct {
		v int
	}
	for _, v := range s {
		go func() {
			_ = T{v: 1}
			_ = map[int]int{v: 1} // want "loop variable v captured by func literal"
		}()
	}

	// ordinary for-loops
	for i := 0; i < 10; i++ {
		go func() {
			print(i) // want "loop variable i captured by func literal"
		}()
	}
	for i, j := 0, 1; i < 100; i, j = j, i+j {
		go func() {
			print(j) // want "loop variable j captured by func literal"
		}()
	}
	type cons struct {
		car int
		cdr *cons
	}
	var head *cons
	for p := head; p != nil; p = p.cdr {
		go func() {
			print(p.car) // want "loop variable p captured by func literal"
		}()
	}
}

// Cases that rely on recursively checking for last statements.
func _() {

	for i := range "outer" {
		for j := range "inner" {
			if j < 1 {
				defer func() {
					print(i) // want "loop variable i captured by func literal"
				}()
			} else if j < 2 {
				go func() {
					print(i) // want "loop variable i captured by func literal"
				}()
			} else {
				go func() {
					print(i)
				}()
				println("we don't catch the error above because of this statement")
			}
		}
	}

	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			if j < 1 {
				switch j {
				case 0:
					defer func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				default:
					go func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				}
			} else if j < 2 {
				var a interface{} = j
				switch a.(type) {
				case int:
					defer func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				default:
					go func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				}
			} else {
				ch := make(chan string)
				select {
				case <-ch:
					defer func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				default:
					go func() {
						print(i) // want "loop variable i captured by func literal"
					}()
				}
			}
		}
	}
}

// Group is used to test that loopclosure only matches Group.Go when Group is
// from the golang.org/x/sync/errgroup package.
type Group struct{}

func (g *Group) Go(func() error) {}

func _() {
	var s []int
	// errgroup.Group.Go() invokes Go routines
	g := new(errgroup.Group)
	for i, v := range s {
		g.Go(func() error {
			print(i) // want "loop variable i captured by func literal"
			print(v) // want "loop variable v captured by func literal"
			return nil
		})
	}

	for i, v := range s {
		if i > 0 {
			g.Go(func() error {
				print(i) // want "loop variable i captured by func literal"
				return nil
			})
		} else {
			g.Go(func() error {
				print(v) // want "loop variable v captured by func literal"
				return nil
			})
		}
	}

	// Do not match other Group.Go cases
	g1 := new(Group)
	for i, v := range s {
		g1.Go(func() error {
			print(i)
			print(v)
			return nil
		})
	}
}

// Real-world example from #16520, slightly simplified
func _() {
	var nodes []interface{}

	critical := new(errgroup.Group)
	others := sync.WaitGroup{}

	isCritical := func(node interface{}) bool { return false }
	run := func(node interface{}) error { return nil }

	for _, node := range nodes {
		if isCritical(node) {
			critical.Go(func() error {
				return run(node) // want "loop variable node captured by func literal"
			})
		} else {
			others.Add(1)
			go func() {
				_ = run(node) // want "loop variable node captured by func literal"
				others.Done()
			}()
		}
	}
}
