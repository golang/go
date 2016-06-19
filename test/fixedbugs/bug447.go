// runoutput

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3804
// test all possible float -> integer conversions

package main

import (
	"bytes"
	"fmt"
	"strings"
)

var (
	intWidths = []int{8, 16, 32, 64} // int%d and uint%d
	floatWidths = []int{32, 64} // float%d
)

func main() {

	var names, funcs bytes.Buffer

	for _, iWidth := range intWidths {
		for _, typ := range []string{"int", "uint"} {
			var segs bytes.Buffer
			itype := fmt.Sprintf("%s%d", typ, iWidth)
			names.WriteString("\ttest" + itype + ",\n")
			for _, fWidth := range floatWidths {
				ftype := fmt.Sprintf("float%d", fWidth)
				seg := strings.Replace(testSegment, "$F", ftype, -1)
				seg = strings.Replace(seg, "$I", itype, -1)
				segs.WriteString(seg)
			}
			body := strings.Replace(testFunc, "$I", itype, -1)
			if typ[0] == 'u' {
				body = strings.Replace(body, "$TEST", " || i < 0", 1)
			} else {
				body = strings.Replace(body, "$TEST", "", 1)
			}
			body = strings.Replace(body, "$TESTSEGMENTS", segs.String(), 1)
			funcs.WriteString(body)
		}
	}

	program = strings.Replace(program, "$NAMES", names.String(), 1)
	program = strings.Replace(program, "$FUNCS", funcs.String(), 1)
	fmt.Print(program)
}

const testSegment = `
	f$F := $F(f)
	if math.Abs(float64(f$F) - f) < 0.05 {
		if v := $I(f$F); v != $I(i) {
			fmt.Printf("$I($F(%f)) = %v, expected %v\n", f, v, i)
		}
	}`

const testFunc =
`func test$I(f float64, i int64) {
	if i != int64($I(i))$TEST {
		return
	}
$TESTSEGMENTS
}
`

var program =
`package main

import (
	"fmt"
	"math"
)

var tests = []struct {
	f float64
	i int64
}{
	{39.7, 39},
	{-39.7, -39},
	{258.6, 258},
	{-258.6, -258},
	{65538.9, 65538},
	{-65538.9, -65538},
	{4294967298.8, 4294967298},
	{-4294967298.8, -4294967298},
}

var funcs = []func(float64, int64){
$NAMES
}

$FUNCS
func main() {
	for _, t := range tests {
		for _, f := range funcs {
			f(t.f, t.i)
		}
	}
}
`
