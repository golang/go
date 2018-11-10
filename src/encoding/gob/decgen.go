// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// encgen writes the helper functions for encoding. Intended to be
// used with go generate; see the invocation in encode.go.

// TODO: We could do more by being unsafe. Add a -unsafe flag?

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
)

var output = flag.String("output", "dec_helpers.go", "file name to write")

type Type struct {
	lower   string
	upper   string
	decoder string
}

var types = []Type{
	{
		"bool",
		"Bool",
		`slice[i] = state.decodeUint() != 0`,
	},
	{
		"complex64",
		"Complex64",
		`real := float32FromBits(state.decodeUint(), ovfl)
		imag := float32FromBits(state.decodeUint(), ovfl)
		slice[i] = complex(float32(real), float32(imag))`,
	},
	{
		"complex128",
		"Complex128",
		`real := float64FromBits(state.decodeUint())
		imag := float64FromBits(state.decodeUint())
		slice[i] = complex(real, imag)`,
	},
	{
		"float32",
		"Float32",
		`slice[i] = float32(float32FromBits(state.decodeUint(), ovfl))`,
	},
	{
		"float64",
		"Float64",
		`slice[i] = float64FromBits(state.decodeUint())`,
	},
	{
		"int",
		"Int",
		`x := state.decodeInt()
		// MinInt and MaxInt
		if x < ^int64(^uint(0)>>1) || int64(^uint(0)>>1) < x {
			error_(ovfl)
		}
		slice[i] = int(x)`,
	},
	{
		"int16",
		"Int16",
		`x := state.decodeInt()
		if x < math.MinInt16 || math.MaxInt16 < x {
			error_(ovfl)
		}
		slice[i] = int16(x)`,
	},
	{
		"int32",
		"Int32",
		`x := state.decodeInt()
		if x < math.MinInt32 || math.MaxInt32 < x {
			error_(ovfl)
		}
		slice[i] = int32(x)`,
	},
	{
		"int64",
		"Int64",
		`slice[i] = state.decodeInt()`,
	},
	{
		"int8",
		"Int8",
		`x := state.decodeInt()
		if x < math.MinInt8 || math.MaxInt8 < x {
			error_(ovfl)
		}
		slice[i] = int8(x)`,
	},
	{
		"string",
		"String",
		`u := state.decodeUint()
		n := int(u)
		if n < 0 || uint64(n) != u || n > state.b.Len() {
			errorf("length of string exceeds input size (%d bytes)", u)
		}
		if n > state.b.Len() {
			errorf("string data too long for buffer: %d", n)
		}
		// Read the data.
		data := state.b.Bytes()
		if len(data) < n {
			errorf("invalid string length %d: exceeds input size %d", n, len(data))
		}
		slice[i] = string(data[:n])
		state.b.Drop(n)`,
	},
	{
		"uint",
		"Uint",
		`x := state.decodeUint()
		/*TODO if math.MaxUint32 < x {
			error_(ovfl)
		}*/
		slice[i] = uint(x)`,
	},
	{
		"uint16",
		"Uint16",
		`x := state.decodeUint()
		if math.MaxUint16 < x {
			error_(ovfl)
		}
		slice[i] = uint16(x)`,
	},
	{
		"uint32",
		"Uint32",
		`x := state.decodeUint()
		if math.MaxUint32 < x {
			error_(ovfl)
		}
		slice[i] = uint32(x)`,
	},
	{
		"uint64",
		"Uint64",
		`slice[i] = state.decodeUint()`,
	},
	{
		"uintptr",
		"Uintptr",
		`x := state.decodeUint()
		if uint64(^uintptr(0)) < x {
			error_(ovfl)
		}
		slice[i] = uintptr(x)`,
	},
	// uint8 Handled separately.
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("decgen: ")
	flag.Parse()
	if flag.NArg() != 0 {
		log.Fatal("usage: decgen [--output filename]")
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "// Created by decgen --output %s; DO NOT EDIT\n", *output)
	fmt.Fprint(&b, header)
	printMaps(&b, "Array")
	fmt.Fprint(&b, "\n")
	printMaps(&b, "Slice")
	for _, t := range types {
		fmt.Fprintf(&b, arrayHelper, t.lower, t.upper)
		fmt.Fprintf(&b, sliceHelper, t.lower, t.upper, t.decoder)
	}
	source, err := format.Source(b.Bytes())
	if err != nil {
		log.Fatal("source format error:", err)
	}
	fd, err := os.Create(*output)
	_, err = fd.Write(source)
	if err != nil {
		log.Fatal(err)
	}
}

func printMaps(b *bytes.Buffer, upperClass string) {
	fmt.Fprintf(b, "var dec%sHelper = map[reflect.Kind]decHelper{\n", upperClass)
	for _, t := range types {
		fmt.Fprintf(b, "reflect.%s: dec%s%s,\n", t.upper, t.upper, upperClass)
	}
	fmt.Fprintf(b, "}\n")
}

const header = `
// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"math"
	"reflect"
)

`

const arrayHelper = `
func dec%[2]sArray(state *decoderState, v reflect.Value, length int, ovfl error) bool {
	// Can only slice if it is addressable.
	if !v.CanAddr() {
		return false
	}
	return dec%[2]sSlice(state, v.Slice(0, v.Len()), length, ovfl)
}
`

const sliceHelper = `
func dec%[2]sSlice(state *decoderState, v reflect.Value, length int, ovfl error) bool {
	slice, ok := v.Interface().([]%[1]s)
	if !ok {
		// It is kind %[1]s but not type %[1]s. TODO: We can handle this unsafely.
		return false
	}
	for i := 0; i < length; i++ {
		if state.b.Len() == 0 {
			errorf("decoding %[1]s array or slice: length exceeds input size (%%d elements)", length)
		}
		%[3]s
	}
	return true
}
`
