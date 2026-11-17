//go:build ignore

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func next(seed *uint64) uint64 {
	*seed ^= *seed << 13
	*seed ^= *seed >> 7
	*seed ^= *seed << 17
	return *seed
}

func fname(prefix string, c int) string {
	if c < 0 {
		return fmt.Sprintf("%sm%d", prefix, -c)
	}
	if c == 0 {
		return prefix + "z0"
	}
	return fmt.Sprintf("%sp%d", prefix, c)
}

func main() {
	seed := uint64(0x243f6a8885a308d3)
	inputs := []uint64{0, 1, 1 << 63, ^uint64(0), 0x0123456789abcdef}
	for len(inputs) < 69 {
		inputs = append(inputs, next(&seed))
	}
	f, err := os.Create("main.go")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintln(f, "// Copyright 2026 The Go Authors. All rights reserved.")
	fmt.Fprintln(f, "// Use of this source code is governed by a BSD-style")
	fmt.Fprintln(f, "// license that can be found in the LICENSE file.")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "package main")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "import (")
	fmt.Fprintln(f, "\t\"fmt\"")
	fmt.Fprintln(f, "\t\"math/bits\"")
	fmt.Fprintln(f, "\t\"os\"")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "var inputs = [...]uint64{")
	for _, x := range inputs {
		fmt.Fprintf(f, "\t%d,\n", x)
	}
	fmt.Fprintln(f, "}")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "var failed bool")
	fmt.Fprintln(f, "var nlaw int")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "func report(name string, x, got, want uint64) {")
	fmt.Fprintln(f, "\tif nlaw >= 10 {")
	fmt.Fprintln(f, "\t\treturn")
	fmt.Fprintln(f, "\t}")
	fmt.Fprintln(f, "\tnlaw++")
	fmt.Fprintln(f, "\tfmt.Printf(\"LAW %s x=%016x got=%016x want=%016x\\n\", name, x, got, want)")
	fmt.Fprintln(f, "\tfailed = true")
	fmt.Fprintln(f, "}")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "func refRotL64(x uint64, c int) uint64 {")
	fmt.Fprintln(f, "\tc &= 63")
	fmt.Fprintln(f, "\tvar r uint64")
	fmt.Fprintln(f, "\tfor i := 0; i < 64; i++ {")
	fmt.Fprintln(f, "\t\tif x&(uint64(1)<<uint(i)) != 0 {")
	fmt.Fprintln(f, "\t\t\tr |= uint64(1) << uint((i+c)&63)")
	fmt.Fprintln(f, "\t\t}")
	fmt.Fprintln(f, "\t}")
	fmt.Fprintln(f, "\treturn r")
	fmt.Fprintln(f, "}")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "func refRotL32(x uint32, c int) uint32 {")
	fmt.Fprintln(f, "\tc &= 31")
	fmt.Fprintln(f, "\tvar r uint32")
	fmt.Fprintln(f, "\tfor i := 0; i < 32; i++ {")
	fmt.Fprintln(f, "\t\tif x&(uint32(1)<<uint(i)) != 0 {")
	fmt.Fprintln(f, "\t\t\tr |= uint32(1) << uint((i+c)&31)")
	fmt.Fprintln(f, "\t\t}")
	fmt.Fprintln(f, "\t}")
	fmt.Fprintln(f, "\treturn r")
	fmt.Fprintln(f, "}")
	for c := -64; c <= 64; c++ {
		nm := fname("c64", c)
		fmt.Fprintf(f, "\nfunc %s() uint64 {\n", nm)
		fmt.Fprintf(f, "\tconst c = %d\n", c)
		fmt.Fprintln(f, "\tvar sum uint64")
		fmt.Fprintln(f, "\tfor _, x := range inputs {")
		fmt.Fprintln(f, "\t\tgot := bits.RotateLeft64(x, c)")
		fmt.Fprintln(f, "\t\tif want := refRotL64(x, c); got != want {")
		fmt.Fprintf(f, "\t\t\treport(\"rot64:%d\", x, got, want)\n", c)
		fmt.Fprintln(f, "\t\t}")
		fmt.Fprintln(f, "\t\tsum ^= got + 0x9e3779b97f4a7c15")
		fmt.Fprintln(f, "\t}")
		fmt.Fprintln(f, "\treturn sum")
		fmt.Fprintln(f, "}")
		nm = fname("c32", c)
		fmt.Fprintf(f, "\nfunc %s() uint64 {\n", nm)
		fmt.Fprintf(f, "\tconst c = %d\n", c)
		fmt.Fprintln(f, "\tvar sum uint64")
		fmt.Fprintln(f, "\tfor _, x := range inputs {")
		fmt.Fprintln(f, "\t\tv := uint32(x)")
		fmt.Fprintln(f, "\t\tgot := bits.RotateLeft32(v, c)")
		fmt.Fprintln(f, "\t\tif want := refRotL32(v, c); got != want {")
		fmt.Fprintf(f, "\t\t\treport(\"rot32:%d\", x, uint64(got), uint64(want))\n", c)
		fmt.Fprintln(f, "\t\t}")
		fmt.Fprintln(f, "\t\tif uint64(got)>>32 != 0 {")
		fmt.Fprintf(f, "\t\t\treport(\"upper32:%d\", x, uint64(got), uint64(uint32(got)))\n", c)
		fmt.Fprintln(f, "\t\t}")
		fmt.Fprintln(f, "\t\tsum ^= uint64(got) + 0x9e3779b97f4a7c15")
		fmt.Fprintln(f, "\t}")
		fmt.Fprintln(f, "\treturn sum")
		fmt.Fprintln(f, "}")
	}
	for c := 0; c <= 64; c++ {
		nm := fname("i64", c)
		fmt.Fprintf(f, "\nfunc %s() uint64 {\n", nm)
		fmt.Fprintln(f, "\tvar sum uint64")
		fmt.Fprintln(f, "\tfor _, x := range inputs {")
		fmt.Fprintf(f, "\t\tgot := x<<%d | x>>%d\n", c, 64-c)
		fmt.Fprintf(f, "\t\tif want := refRotL64(x, %d); got != want {\n", c)
		fmt.Fprintf(f, "\t\t\treport(\"idiom64:%d\", x, got, want)\n", c)
		fmt.Fprintln(f, "\t\t}")
		fmt.Fprintln(f, "\t\tsum ^= got + 0x9e3779b97f4a7c15")
		fmt.Fprintln(f, "\t}")
		fmt.Fprintln(f, "\treturn sum")
		fmt.Fprintln(f, "}")
	}
	for c := 0; c <= 32; c++ {
		nm := fname("i32", c)
		fmt.Fprintf(f, "\nfunc %s() uint64 {\n", nm)
		fmt.Fprintln(f, "\tvar sum uint64")
		fmt.Fprintln(f, "\tfor _, x := range inputs {")
		fmt.Fprintln(f, "\t\tv := uint32(x)")
		fmt.Fprintf(f, "\t\tgot := v<<%d | v>>%d\n", c, 32-c)
		fmt.Fprintf(f, "\t\tif want := refRotL32(v, %d); got != want {\n", c)
		fmt.Fprintf(f, "\t\t\treport(\"idiom32:%d\", x, uint64(got), uint64(want))\n", c)
		fmt.Fprintln(f, "\t\t}")
		fmt.Fprintln(f, "\t\tsum ^= uint64(got) + 0x9e3779b97f4a7c15")
		fmt.Fprintln(f, "\t}")
		fmt.Fprintln(f, "\treturn sum")
		fmt.Fprintln(f, "}")
	}
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "func main() {")
	for c := -64; c <= 64; c++ {
		fmt.Fprintf(f, "\tfmt.Printf(\"rot64 c=%%d sum=%%016x\\n\", %d, %s())\n", c, fname("c64", c))
	}
	for c := -64; c <= 64; c++ {
		fmt.Fprintf(f, "\tfmt.Printf(\"rot32 c=%%d sum=%%016x\\n\", %d, %s())\n", c, fname("c32", c))
	}
	for c := 0; c <= 64; c++ {
		nm := fname("i64", c)
		if c == 0 {
			nm = "i64z0"
		}
		fmt.Fprintf(f, "\tfmt.Printf(\"idiom64 c=%%d sum=%%016x\\n\", %d, %s())\n", c, nm)
	}
	for c := 0; c <= 32; c++ {
		nm := fname("i32", c)
		if c == 0 {
			nm = "i32z0"
		}
		fmt.Fprintf(f, "\tfmt.Printf(\"idiom32 c=%%d sum=%%016x\\n\", %d, %s())\n", c, nm)
	}
	fmt.Fprintln(f, "\tif failed {")
	fmt.Fprintln(f, "\t\tos.Exit(1)")
	fmt.Fprintln(f, "\t}")
	fmt.Fprintln(f, "\tfmt.Println(\"ok\")")
	fmt.Fprintln(f, "}")
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
