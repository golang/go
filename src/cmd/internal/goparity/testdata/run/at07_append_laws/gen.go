//go:build ignore

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"strings"
)

type shape struct {
	typ  string
	elem string
	size int
	ptr  bool
}

func freeSizes() []int {
	s := make([]int, 0, 130)
	for e := 1; e <= 128; e++ {
		s = append(s, e)
	}
	return append(s, 256, 1024)
}

func ptrSizes() []int {
	return []int{8, 32, 64, 128, 256, 1024}
}

func valFn(sh shape, b *strings.Builder) {
	if sh.ptr {
		fmt.Fprintf(b, "func val%s(j int) %s {\n", sh.typ, sh.typ)
		fmt.Fprintf(b, "\tn := j*1000 + 7\n")
		fmt.Fprintf(b, "\tvar v %s\n", sh.typ)
		fmt.Fprintf(b, "\tv.P = &n\n")
		fmt.Fprintf(b, "\tfor k := range v.B {\n")
		fmt.Fprintf(b, "\t\tv.B[k] = byte(j*31 + k*7 + 1)\n")
		fmt.Fprintf(b, "\t}\n")
		fmt.Fprintf(b, "\treturn v\n")
		fmt.Fprintf(b, "}\n")
		fmt.Fprintf(b, "func flip%s(v %s) %s {\n", sh.typ, sh.typ, sh.typ)
		fmt.Fprintf(b, "\tn := ^*v.P\n")
		fmt.Fprintf(b, "\tv.P = &n\n")
		fmt.Fprintf(b, "\tfor k := range v.B {\n")
		fmt.Fprintf(b, "\t\tv.B[k] = ^v.B[k]\n")
		fmt.Fprintf(b, "\t}\n")
		fmt.Fprintf(b, "\treturn v\n")
		fmt.Fprintf(b, "}\n")
		fmt.Fprintf(b, "func eq%s(a, b %s) bool {\n", sh.typ, sh.typ)
		fmt.Fprintf(b, "\treturn *a.P == *b.P && a.B == b.B\n")
		fmt.Fprintf(b, "}\n")
		return
	}
	fmt.Fprintf(b, "func val%s(j int) %s {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\tvar v %s\n", sh.typ)
	fmt.Fprintf(b, "\tfor k := range v {\n")
	fmt.Fprintf(b, "\t\tv[k] = byte(j*31 + k*7 + 1)\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\treturn v\n")
	fmt.Fprintf(b, "}\n")
	fmt.Fprintf(b, "func flip%s(v %s) %s {\n", sh.typ, sh.typ, sh.typ)
	fmt.Fprintf(b, "\tfor k := range v {\n")
	fmt.Fprintf(b, "\t\tv[k] = ^v[k]\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\treturn v\n")
	fmt.Fprintf(b, "}\n")
	fmt.Fprintf(b, "func eq%s(a, b %s) bool {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\treturn a == b\n")
	fmt.Fprintf(b, "}\n")
}

func testFn(sh shape, b *strings.Builder) {
	es := sh.size
	l1 := 256 / es
	if l1 < 1 {
		l1 = 1
	}
	l2 := 4096 / es
	if l2 < 1 {
		l2 = 1
	}
	fmt.Fprintf(b, "func mk%s(n int) []%s {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\treturn make([]%s, n)\n", sh.typ)
	fmt.Fprintf(b, "}\n")
	fmt.Fprintf(b, "func t%s() {\n", sh.typ)
	fmt.Fprintf(b, "\tvar guardA uint64 = 0x9E3779B97F4A7C15\n")
	fmt.Fprintf(b, "\ts := append([]%s(nil), val%s(0), val%s(1))\n", sh.typ, sh.typ, sh.typ)
	fmt.Fprintf(b, "\tc0 := cap(s)\n")
	fmt.Fprintf(b, "\tt := append(s, val%s(2))\n", sh.typ)
	fmt.Fprintf(b, "\tif c0 > len(s) {\n")
	fmt.Fprintf(b, "\t\tif len(t) != 3 || &t[0] != &s[0] {\n")
	fmt.Fprintf(b, "\t\t\tfail(%q, %d)\n", "alias share "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t} else {\n")
	fmt.Fprintf(b, "\t\told := s[0]\n")
	fmt.Fprintf(b, "\t\tfl := flip%s(s[0])\n", sh.typ)
	fmt.Fprintf(b, "\t\tt[0] = fl\n")
	fmt.Fprintf(b, "\t\tif !eq%s(s[0], old) {\n", sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d)\n", "alias leak "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t\tif !eq%s(t[0], fl) {\n", sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d)\n", "alias write "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t\tt[0] = val%s(0)\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range t {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(t[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content append "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tb := []%s(nil)\n", sh.typ)
	fmt.Fprintf(b, "\tfor j := range %d {\n", l1)
	fmt.Fprintf(b, "\t\tb = append(b, val%s(j))\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tb = append(b, val%s(%d), val%s(%d))\n", sh.typ, l1, sh.typ, l1+1)
	fmt.Fprintf(b, "\tif len(b) != %d {\n", l1+2)
	fmt.Fprintf(b, "\t\tfail(%q, %d, len(b))\n", "length crossing256 "+sh.typ, es)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range b {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(b[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content crossing256 "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tc := []%s(nil)\n", sh.typ)
	fmt.Fprintf(b, "\tfor j := range %d {\n", l2)
	fmt.Fprintf(b, "\t\tc = append(c, val%s(j))\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tc = append(c, val%s(%d), val%s(%d), val%s(%d))\n", sh.typ, l2, sh.typ, l2+1, sh.typ, l2+2)
	fmt.Fprintf(b, "\tif len(c) != %d {\n", l2+3)
	fmt.Fprintf(b, "\t\tfail(%q, %d, len(c))\n", "length crossing4096 "+sh.typ, es)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range c {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(c[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content crossing4096 "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\ttmp := make([]%s, %d)\n", sh.typ, l2+3)
	fmt.Fprintf(b, "\tfor j := range tmp {\n")
	fmt.Fprintf(b, "\t\ttmp[j] = val%s(%d + j)\n", sh.typ, l2)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\td := append([]%s(nil), tmp...)\n", sh.typ)
	fmt.Fprintf(b, "\tif len(d) != %d {\n", l2+3)
	fmt.Fprintf(b, "\t\tfail(%q, %d, len(d))\n", "length spread "+sh.typ, es)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range d {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(d[j], val%s(%d+j)) {\n", sh.typ, sh.typ, l2)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content spread "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tvar e []%s\n", sh.typ)
	fmt.Fprintf(b, "\tfor j := range %d {\n", l2+10)
	fmt.Fprintf(b, "\t\te = append(e, val%s(j))\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range e {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(e[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content singles "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tm := mk%s(dynK + %d)\n", sh.typ, l2+3)
	fmt.Fprintf(b, "\tfor j := range m {\n")
	fmt.Fprintf(b, "\t\tm[j] = val%s(j)\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tclobberBytes(len(m) * %d)\n", es)
	fmt.Fprintf(b, "\tfor j := range m {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(m[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content returned "+sh.typ, es)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tif cap(s) < len(s) || cap(t) < len(t) || cap(b) < len(b) || cap(c) < len(c) || cap(d) < len(d) || cap(e) < len(e) || cap(m) < len(m) {\n")
	fmt.Fprintf(b, "\t\tfail(%q, %d)\n", "cap invariant "+sh.typ, es)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tvar guardB uint64 = 0xBF58476D1CE4E5B9\n")
	fmt.Fprintf(b, "\tif guardA != 0x9E3779B97F4A7C15 || guardB != 0xBF58476D1CE4E5B9 {\n")
	fmt.Fprintf(b, "\t\tfail(%q, %d)\n", "canary "+sh.typ, es)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "}\n")
}

func minFn(sh shape, b *strings.Builder) {
	fmt.Fprintf(b, "func u%s() {\n", sh.typ)
	fmt.Fprintf(b, "\tvar guardA uint64 = 0x9E3779B97F4A7C15\n")
	fmt.Fprintf(b, "\tvar e []%s\n", sh.typ)
	fmt.Fprintf(b, "\tfor j := range %d {\n", 4096/sh.size+10)
	fmt.Fprintf(b, "\t\te = append(e, val%s(j))\n", sh.typ)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tfor j := range e {\n")
	fmt.Fprintf(b, "\t\tif !eq%s(e[j], val%s(j)) {\n", sh.typ, sh.typ)
	fmt.Fprintf(b, "\t\t\tfail(%q, %d, j)\n", "content minimal "+sh.typ, sh.size)
	fmt.Fprintf(b, "\t\t}\n")
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tif cap(e) < len(e) {\n")
	fmt.Fprintf(b, "\t\tfail(%q, %d)\n", "cap minimal "+sh.typ, sh.size)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "\tvar guardB uint64 = 0xBF58476D1CE4E5B9\n")
	fmt.Fprintf(b, "\tif guardA != 0x9E3779B97F4A7C15 || guardB != 0xBF58476D1CE4E5B9 {\n")
	fmt.Fprintf(b, "\t\tfail(%q, %d)\n", "canary minimal "+sh.typ, sh.size)
	fmt.Fprintf(b, "\t}\n")
	fmt.Fprintf(b, "}\n")
	fmt.Fprintf(b, "\n")
}

func main() {
	var b strings.Builder
	b.WriteString("// Copyright 2026 The Go Authors. All rights reserved.\n")
	b.WriteString("// Use of this source code is governed by a BSD-style\n")
	b.WriteString("// license that can be found in the LICENSE file.\n\n")
	b.WriteString("package main\n\n")
	b.WriteString("import (\n\t\"fmt\"\n\t\"os\"\n)\n\n")
	b.WriteString("var dynK = 5\n\n")
	b.WriteString("var failures int\n\n")
	b.WriteString("func fail(s string, args ...any) {\n")
	b.WriteString("\tfailures++\n")
	b.WriteString("\tfmt.Print(\"LAW \", s)\n")
	b.WriteString("\tfor _, a := range args {\n")
	b.WriteString("\t\tfmt.Print(\" \", a)\n")
	b.WriteString("\t}\n")
	b.WriteString("\tfmt.Println()\n")
	b.WriteString("}\n\n")
	b.WriteString("func clobberBytes(n int) {\n")
	b.WriteString("\tc := make([]byte, n)\n")
	b.WriteString("\tfor i := range c {\n")
	b.WriteString("\t\tc[i] = 0x55\n")
	b.WriteString("\t}\n")
	b.WriteString("\tvar sum uint64\n")
	b.WriteString("\tfor _, v := range c {\n")
	b.WriteString("\t\tsum += uint64(v)\n")
	b.WriteString("\t}\n")
	b.WriteString("\tif sum != uint64(n)*0x55 {\n")
	b.WriteString("\t\tfail(\"clobber\", n, sum)\n")
	b.WriteString("\t}\n")
	b.WriteString("}\n\n")
	var shapes []shape
	for _, e := range freeSizes() {
		shapes = append(shapes, shape{typ: fmt.Sprintf("e%03d", e), elem: fmt.Sprintf("[%d]byte", e), size: e})
	}
	for _, e := range ptrSizes() {
		shapes = append(shapes, shape{typ: fmt.Sprintf("p%03d", e), elem: fmt.Sprintf("struct{P *int; B [%d]byte}", e), size: 8 + e, ptr: true})
	}
	for _, sh := range shapes {
		if sh.ptr {
			fmt.Fprintf(&b, "type %s struct {\n\tP *int\n\tB [%d]byte\n}\n", sh.typ, sh.size-8)
			continue
		}
		fmt.Fprintf(&b, "type %s %s\n", sh.typ, sh.elem)
	}
	b.WriteString("\n")
	for _, sh := range shapes {
		valFn(sh, &b)
		b.WriteString("\n")
	}
	for _, sh := range shapes {
		testFn(sh, &b)
		b.WriteString("\n")
		if sh.ptr {
			minFn(sh, &b)
		}
	}
	b.WriteString("func main() {\n")
	for _, sh := range shapes {
		fmt.Fprintf(&b, "\tt%s()\n", sh.typ)
		if sh.ptr {
			fmt.Fprintf(&b, "\tu%s()\n", sh.typ)
		}
	}
	fmt.Fprintf(&b, "\tif failures != 0 {\n")
	fmt.Fprintf(&b, "\t\tos.Exit(1)\n")
	fmt.Fprintf(&b, "\t}\n")
	fmt.Fprintf(&b, "\tfmt.Println(%q, %d)\n", "append_laws ok", len(shapes))
	b.WriteString("}\n")
	if err := os.WriteFile("main.go", []byte(b.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
