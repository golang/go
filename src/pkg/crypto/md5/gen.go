// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This program generates md5block.go
// Invoke as
//
//	go run gen.go [-full] |gofmt >md5block.go
//
// The -full flag causes the generated code to do a full
// (16x) unrolling instead of a 4x unrolling.

package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"text/template"
)

func main() {
	flag.Parse()

	t := template.Must(template.New("main").Funcs(funcs).Parse(program))
	if err := t.Execute(os.Stdout, data); err != nil {
		log.Fatal(err)
	}
}

type Data struct {
	a, b, c, d string
	Shift1     []int
	Shift2     []int
	Shift3     []int
	Shift4     []int
	Table1     []uint32
	Table2     []uint32
	Table3     []uint32
	Table4     []uint32
	Full       bool
}

var funcs = template.FuncMap{
	"dup":     dup,
	"relabel": relabel,
	"rotate":  rotate,
}

func dup(count int, x []int) []int {
	var out []int
	for i := 0; i < count; i++ {
		out = append(out, x...)
	}
	return out
}

func relabel(s string) string {
	return strings.NewReplacer("a", data.a, "b", data.b, "c", data.c, "d", data.d).Replace(s)
}

func rotate() string {
	data.a, data.b, data.c, data.d = data.d, data.a, data.b, data.c
	return "" // no output
}

func init() {
	flag.BoolVar(&data.Full, "full", false, "complete unrolling")
}

var data = Data{
	a:      "a",
	b:      "b",
	c:      "c",
	d:      "d",
	Shift1: []int{7, 12, 17, 22},
	Shift2: []int{5, 9, 14, 20},
	Shift3: []int{4, 11, 16, 23},
	Shift4: []int{6, 10, 15, 21},

	// table[i] = int((1<<32) * abs(sin(i+1 radians))).
	Table1: []uint32{
		// round 1
		0xd76aa478,
		0xe8c7b756,
		0x242070db,
		0xc1bdceee,
		0xf57c0faf,
		0x4787c62a,
		0xa8304613,
		0xfd469501,
		0x698098d8,
		0x8b44f7af,
		0xffff5bb1,
		0x895cd7be,
		0x6b901122,
		0xfd987193,
		0xa679438e,
		0x49b40821,
	},
	Table2: []uint32{
		// round 2
		0xf61e2562,
		0xc040b340,
		0x265e5a51,
		0xe9b6c7aa,
		0xd62f105d,
		0x2441453,
		0xd8a1e681,
		0xe7d3fbc8,
		0x21e1cde6,
		0xc33707d6,
		0xf4d50d87,
		0x455a14ed,
		0xa9e3e905,
		0xfcefa3f8,
		0x676f02d9,
		0x8d2a4c8a,
	},
	Table3: []uint32{
		// round3
		0xfffa3942,
		0x8771f681,
		0x6d9d6122,
		0xfde5380c,
		0xa4beea44,
		0x4bdecfa9,
		0xf6bb4b60,
		0xbebfbc70,
		0x289b7ec6,
		0xeaa127fa,
		0xd4ef3085,
		0x4881d05,
		0xd9d4d039,
		0xe6db99e5,
		0x1fa27cf8,
		0xc4ac5665,
	},
	Table4: []uint32{
		// round 4
		0xf4292244,
		0x432aff97,
		0xab9423a7,
		0xfc93a039,
		0x655b59c3,
		0x8f0ccc92,
		0xffeff47d,
		0x85845dd1,
		0x6fa87e4f,
		0xfe2ce6e0,
		0xa3014314,
		0x4e0811a1,
		0xf7537e82,
		0xbd3af235,
		0x2ad7d2bb,
		0xeb86d391,
	},
}

var program = `
// DO NOT EDIT.
// Generate with: go run gen.go{{if .Full}} -full{{end}} | gofmt >md5block.go

// +build !amd64,!386,!arm

package md5

import (
	"unsafe"
	"runtime"
)

{{if not .Full}}
	var t1 = [...]uint32{
	{{range .Table1}}{{printf "\t%#x,\n" .}}{{end}}
	}
	
	var t2 = [...]uint32{
	{{range .Table2}}{{printf "\t%#x,\n" .}}{{end}}
	}
	
	var t3 = [...]uint32{
	{{range .Table3}}{{printf "\t%#x,\n" .}}{{end}}
	}
	
	var t4 = [...]uint32{
	{{range .Table4}}{{printf "\t%#x,\n" .}}{{end}}
	}
{{end}}

const x86 = runtime.GOARCH == "amd64" || runtime.GOARCH == "386"

var littleEndian bool

func init() {
	x := uint32(0x04030201)
	y := [4]byte{0x1, 0x2, 0x3, 0x4}
	littleEndian = *(*[4]byte)(unsafe.Pointer(&x)) == y
}

func block(dig *digest, p []byte) {
	a := dig.s[0]
	b := dig.s[1]
	c := dig.s[2]
	d := dig.s[3]
	var X *[16]uint32
	var xbuf [16]uint32
	for len(p) >= chunk {
		aa, bb, cc, dd := a, b, c, d

		// This is a constant condition - it is not evaluated on each iteration.
		if x86 {
			// MD5 was designed so that x86 processors can just iterate
			// over the block data directly as uint32s, and we generate
			// less code and run 1.3x faster if we take advantage of that.
			// My apologies.
			X = (*[16]uint32)(unsafe.Pointer(&p[0]))
		} else if littleEndian && uintptr(unsafe.Pointer(&p[0]))&(unsafe.Alignof(uint32(0))-1) == 0 {
			X = (*[16]uint32)(unsafe.Pointer(&p[0]))
		} else {
			X = &xbuf
			j := 0
			for i := 0; i < 16; i++ {
				X[i&15] = uint32(p[j]) | uint32(p[j+1])<<8 | uint32(p[j+2])<<16 | uint32(p[j+3])<<24
				j += 4
			}
		}

		{{if .Full}}
			// Round 1.
			{{range $i, $s := dup 4 .Shift1}}
				{{index $.Table1 $i | printf "a += (((c^d)&b)^d) + X[%d] + %d" $i | relabel}}
				{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
				{{rotate}}
			{{end}}
	
			// Round 2.
			{{range $i, $s := dup 4 .Shift2}}
				{{index $.Table2 $i | printf "a += (((b^c)&d)^c) + X[(1+5*%d)&15] + %d" $i | relabel}}
				{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
				{{rotate}}
			{{end}}
	
			// Round 3.
			{{range $i, $s := dup 4 .Shift3}}
				{{index $.Table3 $i | printf "a += (b^c^d) + X[(5+3*%d)&15] + %d" $i | relabel}}
				{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
				{{rotate}}
			{{end}}
	
			// Round 4.
			{{range $i, $s := dup 4 .Shift4}}
				{{index $.Table4 $i | printf "a += (c^(b|^d)) + X[(7*%d)&15] + %d" $i | relabel}}
				{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
				{{rotate}}
			{{end}}
		{{else}}
			// Round 1.
			for i := uint(0); i < 16; {
				{{range $s := .Shift1}}
					{{printf "a += (((c^d)&b)^d) + X[i&15] + t1[i&15]" | relabel}}
					{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
					i++
					{{rotate}}
				{{end}}
			}
	
			// Round 2.
			for i := uint(0); i < 16; {
				{{range $s := .Shift2}}
					{{printf "a += (((b^c)&d)^c) + X[(1+5*i)&15] + t2[i&15]" | relabel}}
					{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
					i++
					{{rotate}}
				{{end}}
			}
	
			// Round 3.
			for i := uint(0); i < 16; {
				{{range $s := .Shift3}}
					{{printf "a += (b^c^d) + X[(5+3*i)&15] + t3[i&15]" | relabel}}
					{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
					i++
					{{rotate}}
				{{end}}
			}
	
			// Round 4.
			for i := uint(0); i < 16; {
				{{range $s := .Shift4}}
					{{printf "a += (c^(b|^d)) + X[(7*i)&15] + t4[i&15]" | relabel}}
					{{printf "a = a<<%d | a>>(32-%d) + b" $s $s | relabel}}
					i++
					{{rotate}}
				{{end}}
			}
		{{end}}

		a += aa
		b += bb
		c += cc
		d += dd

		p = p[chunk:]
	}

	dig.s[0] = a
	dig.s[1] = b
	dig.s[2] = c
	dig.s[3] = d
}
`
