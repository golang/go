// runoutput
//go:build !wasm

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"math"
	"math/bits"
	"os"
	"strconv"
	"strings"
)

var types = []string{
	"int",
	"int8",
	"int16",
	"int32",
	"int64",
	"uint",
	"uint8",
	"uint16",
	"uint32",
	"uint64",
	"uintptr",
	"float32",
	"float64",
}

func main() {
	var prog bytes.Buffer
	fmt.Fprintf(&prog, "package main\n\n")
	fmt.Fprintf(&prog, "import ( \"fmt\"; \"math\" )\n")
	for _, t1 := range types {
		for _, t2 := range types {
			fmt.Fprintf(&prog, "func %[1]s_to_%[2]s(x %[1]s) %[2]s { return %[2]s(x) }\n", t1, t2)
		}
	}

	var outputs []string
	var exprs []string

	fmt.Fprintf(&prog, "var (\n")
	for _, t1 := range types {
		var inputs []string
		switch t1 {
		case "int64", "int":
			if t1 == "int64" || bits.UintSize == 64 {
				inputs = append(inputs, "-0x8000_0000_0000_0000", "-0x7fff_ffff_ffff_ffff", "-0x12_3456_7890", "0x12_3456_7890", "0x7fff_ffff_ffff_ffff")
			}
			fallthrough
		case "int32":
			inputs = append(inputs, "-0x8000_0000", "-0x7fff_ffff", "-0x12_3456", "0x12_3456", "0x7fff_ffff")
			fallthrough
		case "int16":
			inputs = append(inputs, "-0x8000", "-0x7fff", "-0x1234", "0x1234", "0x7fff")
			fallthrough
		case "int8":
			inputs = append(inputs, "-0x80", "-0x7f", "-0x12", "-1", "0", "1", "0x12", "0x7f")

		case "uint64", "uint", "uintptr":
			if t1 == "uint64" || bits.UintSize == 64 {
				inputs = append(inputs, "0x12_3456_7890", "0x7fff_ffff_ffff_ffff", "0x8000_0000_0000_0000", "0xffff_ffff_ffff_ffff")
			}
			fallthrough
		case "uint32":
			inputs = append(inputs, "0x12_3456", "0x7fff_ffff", "0x8000_0000", "0xffff_ffff")
			fallthrough
		case "uint16":
			inputs = append(inputs, "0x1234", "0x7fff", "0x8000", "0xffff")
			fallthrough
		case "uint8":
			inputs = append(inputs, "0", "1", "0x12", "0x7f", "0x80", "0xff")

		case "float64":
			inputs = append(inputs,
				"-1.79769313486231570814527423731704356798070e+308",
				"-1e300",
				"-1e100",
				"-1e40",
				"-3.5e38",
				"3.5e38",
				"1e40",
				"1e100",
				"1e300",
				"1.79769313486231570814527423731704356798070e+308")
			fallthrough
		case "float32":
			inputs = append(inputs,
				"-3.40282346638528859811704183484516925440e+38",
				"-1e38",
				"-1.5",
				"-1.401298464324817070923729583289916131280e-45",
				"0",
				"1.401298464324817070923729583289916131280e-45",
				"1.5",
				"1e38",
				"3.40282346638528859811704183484516925440e+38")
		}
		for _, t2 := range types {
			for _, x := range inputs {
				code := fmt.Sprintf("%s_to_%s(%s)", t1, t2, x)
				fmt.Fprintf(&prog, "\tv%d = %s\n", len(outputs), code)
				exprs = append(exprs, code)
				outputs = append(outputs, convert(x, t1, t2))
			}
		}
	}
	fmt.Fprintf(&prog, ")\n\n")
	fmt.Fprintf(&prog, "func main() {\n\tok := true\n")
	for i, out := range outputs {
		fmt.Fprintf(&prog, "\tif v%d != %s { fmt.Println(%q, \"=\", v%d, \"want\", %s); ok = false }\n", i, out, exprs[i], i, out)
	}
	fmt.Fprintf(&prog, "\tif !ok { println(\"FAIL\") }\n")
	fmt.Fprintf(&prog, "}\n")

	os.Stdout.Write(prog.Bytes())
}

func convert(x, t1, t2 string) string {
	if strings.HasPrefix(t1, "int") {
		v, err := strconv.ParseInt(x, 0, 64)
		if err != nil {
			println(x, t1, t2)
			panic(err)
		}
		return convert1(v, t2)
	}
	if strings.HasPrefix(t1, "uint") {
		v, err := strconv.ParseUint(x, 0, 64)
		if err != nil {
			println(x, t1, t2)
			panic(err)
		}
		return convert1(v, t2)
	}
	if strings.HasPrefix(t1, "float") {
		v, err := strconv.ParseFloat(x, 64)
		if err != nil {
			println(x, t1, t2)
			panic(err)
		}
		if t1 == "float32" {
			v = float64(float32(v))
		}
		return convert1(v, t2)
	}
	panic(t1)
}

func convert1[T int64 | uint64 | float64](v T, t2 string) string {
	switch t2 {
	case "int":
		return fmt.Sprintf("%s(%#x)", t2, int(v))
	case "int8":
		return fmt.Sprintf("%s(%#x)", t2, int8(v))
	case "int16":
		return fmt.Sprintf("%s(%#x)", t2, int16(v))
	case "int32":
		return fmt.Sprintf("%s(%#x)", t2, int32(v))
	case "int64":
		return fmt.Sprintf("%s(%#x)", t2, int64(v))
	case "uint":
		return fmt.Sprintf("%s(%#x)", t2, uint(v))
	case "uint8":
		return fmt.Sprintf("%s(%#x)", t2, uint8(v))
	case "uint16":
		return fmt.Sprintf("%s(%#x)", t2, uint16(v))
	case "uint32":
		return fmt.Sprintf("%s(%#x)", t2, uint32(v))
	case "uint64":
		return fmt.Sprintf("%s(%#x)", t2, uint64(v))
	case "uintptr":
		return fmt.Sprintf("%s(%#x)", t2, uintptr(v))
	case "float32":
		v := float32(v)
		if math.IsInf(float64(v), -1) {
			return "float32(math.Inf(-1))"
		}
		if math.IsInf(float64(v), +1) {
			return "float32(math.Inf(+1))"
		}
		return fmt.Sprintf("%s(%v)", t2, float64(v))
	case "float64":
		return fmt.Sprintf("%s(%v)", t2, float64(v))
	}
	panic(t2)
}
