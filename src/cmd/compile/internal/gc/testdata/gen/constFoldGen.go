// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program generates a test to verify that the standard arithmetic
// operators properly handle constant folding. The test file should be
// generated with a known working version of go.
// launch with `go run constFoldGen.go` a file called constFold_test.go
// will be written into the grandparent directory containing the tests.

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
)

type op struct {
	name, symbol string
}
type szD struct {
	name string
	sn   string
	u    []uint64
	i    []int64
}

var szs []szD = []szD{
	szD{name: "uint64", sn: "64", u: []uint64{0, 1, 4294967296, 0xffffFFFFffffFFFF}},
	szD{name: "int64", sn: "64", i: []int64{-0x8000000000000000, -0x7FFFFFFFFFFFFFFF,
		-4294967296, -1, 0, 1, 4294967296, 0x7FFFFFFFFFFFFFFE, 0x7FFFFFFFFFFFFFFF}},

	szD{name: "uint32", sn: "32", u: []uint64{0, 1, 4294967295}},
	szD{name: "int32", sn: "32", i: []int64{-0x80000000, -0x7FFFFFFF, -1, 0,
		1, 0x7FFFFFFF}},

	szD{name: "uint16", sn: "16", u: []uint64{0, 1, 65535}},
	szD{name: "int16", sn: "16", i: []int64{-32768, -32767, -1, 0, 1, 32766, 32767}},

	szD{name: "uint8", sn: "8", u: []uint64{0, 1, 255}},
	szD{name: "int8", sn: "8", i: []int64{-128, -127, -1, 0, 1, 126, 127}},
}

var ops = []op{
	op{"add", "+"}, op{"sub", "-"}, op{"div", "/"}, op{"mul", "*"},
	op{"lsh", "<<"}, op{"rsh", ">>"}, op{"mod", "%"},
}

// compute the result of i op j, cast as type t.
func ansU(i, j uint64, t, op string) string {
	var ans uint64
	switch op {
	case "+":
		ans = i + j
	case "-":
		ans = i - j
	case "*":
		ans = i * j
	case "/":
		if j != 0 {
			ans = i / j
		}
	case "%":
		if j != 0 {
			ans = i % j
		}
	case "<<":
		ans = i << j
	case ">>":
		ans = i >> j
	}
	switch t {
	case "uint32":
		ans = uint64(uint32(ans))
	case "uint16":
		ans = uint64(uint16(ans))
	case "uint8":
		ans = uint64(uint8(ans))
	}
	return fmt.Sprintf("%d", ans)
}

// compute the result of i op j, cast as type t.
func ansS(i, j int64, t, op string) string {
	var ans int64
	switch op {
	case "+":
		ans = i + j
	case "-":
		ans = i - j
	case "*":
		ans = i * j
	case "/":
		if j != 0 {
			ans = i / j
		}
	case "%":
		if j != 0 {
			ans = i % j
		}
	case "<<":
		ans = i << uint64(j)
	case ">>":
		ans = i >> uint64(j)
	}
	switch t {
	case "int32":
		ans = int64(int32(ans))
	case "int16":
		ans = int64(int16(ans))
	case "int8":
		ans = int64(int8(ans))
	}
	return fmt.Sprintf("%d", ans)
}

func main() {

	w := new(bytes.Buffer)

	fmt.Fprintf(w, "package gc\n")
	fmt.Fprintf(w, "import \"testing\"\n")

	for _, s := range szs {
		for _, o := range ops {
			if o.symbol == "<<" || o.symbol == ">>" {
				// shifts handled separately below, as they can have
				// different types on the LHS and RHS.
				continue
			}
			fmt.Fprintf(w, "func TestConstFold%s%s(t *testing.T) {\n", s.name, o.name)
			fmt.Fprintf(w, "\tvar x, y, r %s\n", s.name)
			// unsigned test cases
			for _, c := range s.u {
				fmt.Fprintf(w, "\tx = %d\n", c)
				for _, d := range s.u {
					if d == 0 && (o.symbol == "/" || o.symbol == "%") {
						continue
					}
					fmt.Fprintf(w, "\ty = %d\n", d)
					fmt.Fprintf(w, "\tr = x %s y\n", o.symbol)
					want := ansU(c, d, s.name, o.symbol)
					fmt.Fprintf(w, "\tif r != %s {\n", want)
					fmt.Fprintf(w, "\t\tt.Errorf(\"%d %s %d = %%d, want %s\", r)\n", c, o.symbol, d, want)
					fmt.Fprintf(w, "\t}\n")
				}
			}
			// signed test cases
			for _, c := range s.i {
				fmt.Fprintf(w, "\tx = %d\n", c)
				for _, d := range s.i {
					if d == 0 && (o.symbol == "/" || o.symbol == "%") {
						continue
					}
					fmt.Fprintf(w, "\ty = %d\n", d)
					fmt.Fprintf(w, "\tr = x %s y\n", o.symbol)
					want := ansS(c, d, s.name, o.symbol)
					fmt.Fprintf(w, "\tif r != %s {\n", want)
					fmt.Fprintf(w, "\t\tt.Errorf(\"%d %s %d = %%d, want %s\", r)\n", c, o.symbol, d, want)
					fmt.Fprintf(w, "\t}\n")
				}
			}
			fmt.Fprintf(w, "}\n")
		}
	}

	// Special signed/unsigned cases for shifts
	for _, ls := range szs {
		for _, rs := range szs {
			if rs.name[0] != 'u' {
				continue
			}
			for _, o := range ops {
				if o.symbol != "<<" && o.symbol != ">>" {
					continue
				}
				fmt.Fprintf(w, "func TestConstFold%s%s%s(t *testing.T) {\n", ls.name, rs.name, o.name)
				fmt.Fprintf(w, "\tvar x, r %s\n", ls.name)
				fmt.Fprintf(w, "\tvar y %s\n", rs.name)
				// unsigned LHS
				for _, c := range ls.u {
					fmt.Fprintf(w, "\tx = %d\n", c)
					for _, d := range rs.u {
						fmt.Fprintf(w, "\ty = %d\n", d)
						fmt.Fprintf(w, "\tr = x %s y\n", o.symbol)
						want := ansU(c, d, ls.name, o.symbol)
						fmt.Fprintf(w, "\tif r != %s {\n", want)
						fmt.Fprintf(w, "\t\tt.Errorf(\"%d %s %d = %%d, want %s\", r)\n", c, o.symbol, d, want)
						fmt.Fprintf(w, "\t}\n")
					}
				}
				// signed LHS
				for _, c := range ls.i {
					fmt.Fprintf(w, "\tx = %d\n", c)
					for _, d := range rs.u {
						fmt.Fprintf(w, "\ty = %d\n", d)
						fmt.Fprintf(w, "\tr = x %s y\n", o.symbol)
						want := ansS(c, int64(d), ls.name, o.symbol)
						fmt.Fprintf(w, "\tif r != %s {\n", want)
						fmt.Fprintf(w, "\t\tt.Errorf(\"%d %s %d = %%d, want %s\", r)\n", c, o.symbol, d, want)
						fmt.Fprintf(w, "\t}\n")
					}
				}
				fmt.Fprintf(w, "}\n")
			}
		}
	}

	// Constant folding for comparisons
	for _, s := range szs {
		fmt.Fprintf(w, "func TestConstFoldCompare%s(t *testing.T) {\n", s.name)
		for _, x := range s.i {
			for _, y := range s.i {
				fmt.Fprintf(w, "\t{\n")
				fmt.Fprintf(w, "\t\tvar x %s = %d\n", s.name, x)
				fmt.Fprintf(w, "\t\tvar y %s = %d\n", s.name, y)
				if x == y {
					fmt.Fprintf(w, "\t\tif !(x == y) { t.Errorf(\"!(%%d == %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x == y { t.Errorf(\"%%d == %%d\", x, y) }\n")
				}
				if x != y {
					fmt.Fprintf(w, "\t\tif !(x != y) { t.Errorf(\"!(%%d != %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x != y { t.Errorf(\"%%d != %%d\", x, y) }\n")
				}
				if x < y {
					fmt.Fprintf(w, "\t\tif !(x < y) { t.Errorf(\"!(%%d < %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x < y { t.Errorf(\"%%d < %%d\", x, y) }\n")
				}
				if x > y {
					fmt.Fprintf(w, "\t\tif !(x > y) { t.Errorf(\"!(%%d > %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x > y { t.Errorf(\"%%d > %%d\", x, y) }\n")
				}
				if x <= y {
					fmt.Fprintf(w, "\t\tif !(x <= y) { t.Errorf(\"!(%%d <= %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x <= y { t.Errorf(\"%%d <= %%d\", x, y) }\n")
				}
				if x >= y {
					fmt.Fprintf(w, "\t\tif !(x >= y) { t.Errorf(\"!(%%d >= %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x >= y { t.Errorf(\"%%d >= %%d\", x, y) }\n")
				}
				fmt.Fprintf(w, "\t}\n")
			}
		}
		for _, x := range s.u {
			for _, y := range s.u {
				fmt.Fprintf(w, "\t{\n")
				fmt.Fprintf(w, "\t\tvar x %s = %d\n", s.name, x)
				fmt.Fprintf(w, "\t\tvar y %s = %d\n", s.name, y)
				if x == y {
					fmt.Fprintf(w, "\t\tif !(x == y) { t.Errorf(\"!(%%d == %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x == y { t.Errorf(\"%%d == %%d\", x, y) }\n")
				}
				if x != y {
					fmt.Fprintf(w, "\t\tif !(x != y) { t.Errorf(\"!(%%d != %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x != y { t.Errorf(\"%%d != %%d\", x, y) }\n")
				}
				if x < y {
					fmt.Fprintf(w, "\t\tif !(x < y) { t.Errorf(\"!(%%d < %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x < y { t.Errorf(\"%%d < %%d\", x, y) }\n")
				}
				if x > y {
					fmt.Fprintf(w, "\t\tif !(x > y) { t.Errorf(\"!(%%d > %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x > y { t.Errorf(\"%%d > %%d\", x, y) }\n")
				}
				if x <= y {
					fmt.Fprintf(w, "\t\tif !(x <= y) { t.Errorf(\"!(%%d <= %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x <= y { t.Errorf(\"%%d <= %%d\", x, y) }\n")
				}
				if x >= y {
					fmt.Fprintf(w, "\t\tif !(x >= y) { t.Errorf(\"!(%%d >= %%d)\", x, y) }\n")
				} else {
					fmt.Fprintf(w, "\t\tif x >= y { t.Errorf(\"%%d >= %%d\", x, y) }\n")
				}
				fmt.Fprintf(w, "\t}\n")
			}
		}
		fmt.Fprintf(w, "}\n")
	}

	// gofmt result
	b := w.Bytes()
	src, err := format.Source(b)
	if err != nil {
		fmt.Printf("%s\n", b)
		panic(err)
	}

	// write to file
	err = ioutil.WriteFile("../../constFold_test.go", src, 0666)
	if err != nil {
		log.Fatalf("can't write output: %v\n", err)
	}
}
