// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains examples to embed in the Go 1 release notes document.

package main

import "log"

func main() {
	stringAppend()
	mapDelete()
	mapIteration()
	multipleAssignment()
	structEquality()
}

func mapDelete() {
	m := map[string]int{"7": 7, "23": 23}
	k := "7"
	delete(m, k)
	if m["7"] != 0 || m["23"] != 23 {
		log.Fatal("mapDelete:", m)
	}
}

func stringAppend() {
	greeting := []byte{}
	greeting = append(greeting, []byte("hello ")...)
	greeting = append(greeting, "world"...)
	if string(greeting) != "hello world" {
		log.Fatal("stringAppend: ", string(greeting))
	}
}

func mapIteration() {
	m := map[string]int{"Sunday": 0, "Monday": 1}
	for name, value := range m {
		// This loop should not assume Sunday will be visited first.
		f(name, value)
	}
}

func assert(t bool) {
	if !t {
		log.Panic("assertion fail")
	}
}

func multipleAssignment() {
	sa := []int{1, 2, 3}
	i := 0
	i, sa[i] = 1, 2 // sets i = 1, sa[0] = 2

	sb := []int{1, 2, 3}
	j := 0
	sb[j], j = 2, 1 // sets sb[0] = 2, j = 1

	sc := []int{1, 2, 3}
	sc[0], sc[0] = 1, 2 // sets sc[0] = 1, then sc[0] = 2 (so sc[0] = 2 at end)

	assert(i == 1 && sa[0] == 2)
	assert(j == 1 && sb[0] == 2)
	assert(sc[0] == 2)
}

func structEquality() {
	// Feature not net in repo.
	//	type Day struct {
	//		long string
	//		short string
	//	}
	//	Christmas := Day{"Christmas", "XMas"}
	//	Thanksgiving := Day{"Thanksgiving", "Turkey"}
	//	holiday := map[Day]bool {
	//		Christmas: true,
	//		Thanksgiving: true,
	//	}
	//	fmt.Printf("Christmas is a holiday: %t\n", holiday[Christmas])
}

func f(string, int) {
}

func initializationFunction(c chan int) {
	c <- 1
}

var PackageGlobal int

func init() {
	c := make(chan int)
	go initializationFunction(c)
	PackageGlobal = <-c
}
