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

var output = flag.String("output", "enc_helpers.go", "file name to write")

type Type struct {
	lower   string
	upper   string
	zero    string
	encoder string
}

var types = []Type{
	{
		"bool",
		"Bool",
		"false",
		`if x {
			state.encodeUint(1)
		} else {
			state.encodeUint(0)
		}`,
	},
	{
		"complex64",
		"Complex64",
		"0+0i",
		`rpart := floatBits(float64(real(x)))
		ipart := floatBits(float64(imag(x)))
		state.encodeUint(rpart)
		state.encodeUint(ipart)`,
	},
	{
		"complex128",
		"Complex128",
		"0+0i",
		`rpart := floatBits(real(x))
		ipart := floatBits(imag(x))
		state.encodeUint(rpart)
		state.encodeUint(ipart)`,
	},
	{
		"float32",
		"Float32",
		"0",
		`bits := floatBits(float64(x))
		state.encodeUint(bits)`,
	},
	{
		"float64",
		"Float64",
		"0",
		`bits := floatBits(x)
		state.encodeUint(bits)`,
	},
	{
		"int",
		"Int",
		"0",
		`state.encodeInt(int64(x))`,
	},
	{
		"int16",
		"Int16",
		"0",
		`state.encodeInt(int64(x))`,
	},
	{
		"int32",
		"Int32",
		"0",
		`state.encodeInt(int64(x))`,
	},
	{
		"int64",
		"Int64",
		"0",
		`state.encodeInt(x)`,
	},
	{
		"int8",
		"Int8",
		"0",
		`state.encodeInt(int64(x))`,
	},
	{
		"string",
		"String",
		`""`,
		`state.encodeUint(uint64(len(x)))
		state.b.WriteString(x)`,
	},
	{
		"uint",
		"Uint",
		"0",
		`state.encodeUint(uint64(x))`,
	},
	{
		"uint16",
		"Uint16",
		"0",
		`state.encodeUint(uint64(x))`,
	},
	{
		"uint32",
		"Uint32",
		"0",
		`state.encodeUint(uint64(x))`,
	},
	{
		"uint64",
		"Uint64",
		"0",
		`state.encodeUint(x)`,
	},
	{
		"uintptr",
		"Uintptr",
		"0",
		`state.encodeUint(uint64(x))`,
	},
	// uint8 Handled separately.
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("encgen: ")
	flag.Parse()
	if flag.NArg() != 0 {
		log.Fatal("usage: encgen [--output filename]")
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "// Created by encgen --output %s; DO NOT EDIT\n", *output)
	fmt.Fprint(&b, header)
	printMaps(&b, "Array")
	fmt.Fprint(&b, "\n")
	printMaps(&b, "Slice")
	for _, t := range types {
		fmt.Fprintf(&b, arrayHelper, t.lower, t.upper)
		fmt.Fprintf(&b, sliceHelper, t.lower, t.upper, t.zero, t.encoder)
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
	fmt.Fprintf(b, "var enc%sHelper = map[reflect.Kind]encHelper{\n", upperClass)
	for _, t := range types {
		fmt.Fprintf(b, "reflect.%s: enc%s%s,\n", t.upper, t.upper, upperClass)
	}
	fmt.Fprintf(b, "}\n")
}

const header = `
// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"reflect"
)

`

const arrayHelper = `
func enc%[2]sArray(state *encoderState, v reflect.Value) bool {
	// Can only slice if it is addressable.
	if !v.CanAddr() {
		return false
	}
	return enc%[2]sSlice(state, v.Slice(0, v.Len()))
}
`

const sliceHelper = `
func enc%[2]sSlice(state *encoderState, v reflect.Value) bool {
	slice, ok := v.Interface().([]%[1]s)
	if !ok {
		// It is kind %[1]s but not type %[1]s. TODO: We can handle this unsafely.
		return false
	}
	for _, x := range slice {
		if x != %[3]s || state.sendZero {
			%[4]s
		}
	}
	return true
}
`
