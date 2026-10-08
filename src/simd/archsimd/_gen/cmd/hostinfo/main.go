// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"

	"simd/archsimd"
)

type row struct {
	name  string
	value string
}

func main() {
	defaultHost := os.Getenv("GO_BUILDER_NAME")
	if defaultHost == "" {
		if h, err := os.Hostname(); err == nil && h != "" {
			defaultHost = h
		} else {
			defaultHost = "host"
		}
	}

	hostFlag := flag.String("host", defaultHost, "host or builder name to display in the table header")
	flag.Parse()

	var archName string
	var rows []row

	switch runtime.GOARCH {
	case "amd64":
		archName = "amd64"
		rows = reflectFeatures(archsimd.X86)
	case "arm64":
		archName = "arm64"
		rows = reflectFeatures(archsimd.ARM64)
		if bits, ok := sveVectorBits(); ok {
			rows = append(rows, row{name: "SVE vector size", value: strconv.Itoa(bits)})
		} else {
			rows = append(rows, row{name: "SVE vector size", value: "n/a"})
		}
	case "wasm":
		archName = "wasm"
		// Smoke check basic SIMD execution
		_ = archsimd.Int32x4{}.Add(archsimd.Int32x4{})
		rows = append(rows, row{name: "SIMD128", value: "✔"})
	default:
		archName = runtime.GOARCH
		rows = append(rows, row{name: "Supported", value: "✘"})
	}

	printTable(archName, *hostFlag, rows)
}

func reflectFeatures(receiver any) []row {
	t := reflect.TypeOf(receiver)
	v := reflect.ValueOf(receiver)

	var rows []row
	for i := 0; i < t.NumMethod(); i++ {
		m := t.Method(i)
		res := m.Func.Call([]reflect.Value{v})[0].Bool()
		valStr := "✘"
		if res {
			valStr = "✔"
		}
		rows = append(rows, row{
			name:  m.Name,
			value: valStr,
		})
	}
	return rows
}

func printTable(arch, host string, rows []row) {
	col1Width := len(arch)
	col2Width := len(host)

	for _, r := range rows {
		if len(r.name) > col1Width {
			col1Width = len(r.name)
		}
		if len(r.value) > col2Width {
			col2Width = len(r.value)
		}
	}

	header := fmt.Sprintf("| %-*s | %-*s |", col1Width, arch, col2Width, host)
	sep := fmt.Sprintf("| %s | %s |", strings.Repeat("-", col1Width), strings.Repeat("-", col2Width))

	fmt.Println(header)
	fmt.Println(sep)
	for _, r := range rows {
		fmt.Printf("| %-*s | %-*s |\n", col1Width, r.name, col2Width, r.value)
	}
}
