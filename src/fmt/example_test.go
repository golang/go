// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// The Errorf function lets us use formatting features
// to create descriptive error messages.
func ExampleErrorf() {
	const name, id = "bueller", 17
	err := fmt.Errorf("user %q (id %d) not found", name, id)
	fmt.Println(err.Error())
	// Output: user "bueller" (id 17) not found
}

func ExampleFscanf() {
	var (
		i int
		b bool
		s string
	)
	r := strings.NewReader("5 true gophers")
	n, err := fmt.Fscanf(r, "%d %t %s", &i, &b, &s)
	if err != nil {
		panic(err)
	}
	fmt.Println(i, b, s)
	fmt.Println(n)
	// Output:
	// 5 true gophers
	// 3
}

func ExampleSprintf() {
	i := 30
	s := "Aug"
	sf := fmt.Sprintf("Today is %d %s", i, s)
	fmt.Println(sf)
	fmt.Println(len(sf))
	// Output:
	// Today is 30 Aug
	// 15
}

func ExamplePrint() {
	n, err := fmt.Print("there", "are", 99, "gophers", "\n")
	if err != nil {
		panic(err)
	}
	fmt.Print(n)
	// Output:
	// thereare99gophers
	// 18
}

func ExamplePrintln() {
	n, err := fmt.Println("there", "are", 99, "gophers")
	if err != nil {
		panic(err)
	}
	fmt.Print(n)
	// Output:
	// there are 99 gophers
	// 21
}

func ExampleSprintln() {
	s := "Aug"
	sl := fmt.Sprintln("Today is 30", s)
	fmt.Printf("%q", sl)
	// Output:
	// "Today is 30 Aug\n"
}

func ExampleFprint() {
	n, err := fmt.Fprint(os.Stdout, "there", "are", 99, "gophers", "\n")
	if err != nil {
		panic(err)
	}
	fmt.Print(n)
	// Output:
	// thereare99gophers
	// 18
}

func ExampleFprintln() {
	n, err := fmt.Fprintln(os.Stdout, "there", "are", 99, "gophers")
	if err != nil {
		panic(err)
	}
	fmt.Print(n)
	// Output:
	// there are 99 gophers
	// 21
}

func ExampleFscanln() {
	s := `dmr 1771 1.61803398875
	ken 271828 3.14159`
	r := strings.NewReader(s)
	var a string
	var b int
	var c float64
	for {
		n, err := fmt.Fscanln(r, &a, &b, &c)
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(err)
		}
		fmt.Printf("%d: %s, %d, %f\n", n, a, b, c)
	}
	// Output:
	// 3: dmr, 1771, 1.618034
	// 3: ken, 271828, 3.141590
}

func ExampleSprint() {
	s := fmt.Sprint("there", "are", "99", "gophers")
	fmt.Println(s)
	fmt.Println(len(s))
	// Output:
	// thereare99gophers
	// 17
}

func ExamplePrintf() {
	type point struct {
		x, y int
	}
	p := point{1, 2}

	fmt.Printf("Struct 1: %v\n", p)
	fmt.Printf("Struct 2: %+v\n", p)
	fmt.Printf("Struct 3: %#v\n", p)
	fmt.Printf("Type: %T\n", p)
	fmt.Printf("Pointer: %p\n", &p)

	fmt.Printf("Bool: %t\n", true)
	fmt.Printf("Int 1: %d\n", 123)
	fmt.Printf("Int 2: %c\n", 33)
	fmt.Printf("Binary: %b\n", 14)
	fmt.Printf("Hex: %x\n", 456)
	fmt.Printf("Float 1: %f\n", 78.9)
	fmt.Printf("Float 2: %e\n", 123400000.0)
	fmt.Printf("Float 3: %E\n", 123400000.0)
	fmt.Printf("String 1: %s\n", "\"string\"")
	fmt.Printf("String 2: %q\n", "\"string\"")
	fmt.Printf("String 3: %x\n", "hex this")
	fmt.Printf("String 4: %s\n", "string")

	fmt.Printf("Int numbers 1:|%6d|%6d|\n", 12, 345)
	fmt.Printf("Float numbers 1:|%6.2f|%6.2f|\n", 1.2, 3.45)
	fmt.Printf("Float numbers 2:|%-6.2f|%-6.2f|\n", 1.2, 3.45)
	fmt.Printf("String 5:|%6s|%6s|\n", "foo", "b")
	fmt.Printf("String 6:|%-6s|%-6s|\n", "foo", "b")

	// Output:
	// Struct 1: {1 2}
	// Struct 2: {x:1 y:2}
	// Struct 3: main.point{x:1, y:2}
	// Type: main.point
	// Pointer: 0xc420084010
	// Bool: true
	// Int 1: 123
	// Int 2: !
	// Binary: 1110
	// Hex: 1c8
	// Float 1: 78.900000
	// Float 2: 1.234000e+08
	// Float 3: 1.234000E+08
	// String 1: "string"
	// String 2: "\"string\""
	// String 3: 6865782074686973
	// String 4: string
	// Int numbers 1:|    12|   345|
	// Float numbers 1:|  1.20|  3.45|
	// Float numbers 2:|1.20  |3.45  |
	// String 5:|   foo|     b|
	// String 6:|foo   |b     |
}
