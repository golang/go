// runoutput

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5162: bad array equality when multiple comparisons
// happen in the same expression.

package main

import (
	"fmt"
	"strings"
)

const template = `
func CheckEqNNN_TTT() {
	onesA := [NNN]ttt{ONES}
	onesB := [NNN]ttt{ONES}
	twos := [NNN]ttt{TWOS}
	if onesA != onesB {
		println("onesA != onesB in CheckEqNNN_TTT")
	}
	if onesA == twos {
		println("onesA == twos in CheckEqNNN_TTT")
	}
	if onesB == twos {
		println("onesB == twos in CheckEqNNN_TTT")
	}
	if s := fmt.Sprint(onesA == onesB, onesA != twos, onesB != twos); s != "true true true" {
		println("fail in CheckEqNNN_TTT:", s)
	}
}

func CheckEqNNN_TTTExtraVar() {
	onesA := [NNN]ttt{ONES}
	onesB := [NNN]ttt{ONES}
	twos := [NNN]ttt{TWOS}
	onesX := onesA
	if onesA != onesB {
		println("onesA != onesB in CheckEqNNN_TTTExtraVar")
	}
	if onesA == twos {
		println("onesA == twos in CheckEqNNN_TTTExtraVar")
	}
	if onesB == twos {
		println("onesB == twos in CheckEqNNN_TTTExtraVar")
	}
	if s := fmt.Sprint(onesA == onesB, onesA != twos, onesB != twos); s != "true true true" {
		println("fail in CheckEqNNN_TTTExtraVar:", s)
	}
	if s := fmt.Sprint(onesB == onesX); s != "true" {
		println("extra var fail in CheckEqNNN_TTTExtraVar")
	}
}
`

func main() {
	fmt.Print("// run\n\n")
	fmt.Print("// THIS FILE IS AUTO-GENERATED\n\n")
	fmt.Print("package main\n\n")
	fmt.Println(`import "fmt"`)

	types := []string{
		"int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"float32", "float64"}
	tocall := make([]string, 0, 32*len(types))
	for i := 1; i <= 32; i++ {
		for _, typ := range types {
			src := template
			src = strings.Replace(src, "NNN", fmt.Sprint(i), -1)
			src = strings.Replace(src, "TTT", strings.Title(typ), -1)
			src = strings.Replace(src, "ttt", typ, -1)
			src = strings.Replace(src, "ONES", "1"+strings.Repeat(", 1", i-1), -1)
			src = strings.Replace(src, "TWOS", "2"+strings.Repeat(", 2", i-1), -1)
			fmt.Print(src)
			tocall = append(tocall, fmt.Sprintf("CheckEq%d_%s", i, strings.Title(typ)))
		}
	}
	fmt.Println("func main() {")
	for _, fun := range tocall {
		fmt.Printf("\t%s()\n", fun)
		fmt.Printf("\t%sExtraVar()\n", fun)
	}
	fmt.Println("}")
}
