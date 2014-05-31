// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Flags used by Bench.Measured to indicate
// which measurements a Bench contains.
const (
	NsOp = 1 << iota
	MbS
	BOp
	AllocsOp
)

// Bench is one run of a single benchmark.
type Bench struct {
	Name     string  // benchmark name
	N        int     // number of iterations
	NsOp     float64 // nanoseconds per iteration
	MbS      float64 // MB processed per second
	BOp      uint64  // bytes allocated per iteration
	AllocsOp uint64  // allocs per iteration
	Measured int     // which measurements were recorded
	ord      int     // ordinal position within a benchmark run, used for sorting
}

// ParseLine extracts a Bench from a single line of testing.B output.
func ParseLine(line string) (*Bench, error) {
	fields := strings.Fields(line)

	// Two required, positional fields: Name and iterations.
	if len(fields) < 2 {
		return nil, fmt.Errorf("two fields required, have %d", len(fields))
	}
	if !strings.HasPrefix(fields[0], "Benchmark") {
		return nil, fmt.Errorf(`first field does not start with "Benchmark`)
	}
	n, err := strconv.Atoi(fields[1])
	if err != nil {
		return nil, err
	}
	b := &Bench{Name: fields[0], N: n}

	// Parse any remaining pairs of fields; we've parsed one pair already.
	for i := 1; i < len(fields)/2; i++ {
		b.parseMeasurement(fields[i*2], fields[i*2+1])
	}
	return b, nil
}

func (b *Bench) parseMeasurement(quant string, unit string) {
	switch unit {
	case "ns/op":
		if f, err := strconv.ParseFloat(quant, 64); err == nil {
			b.NsOp = f
			b.Measured |= NsOp
		}
	case "MB/s":
		if f, err := strconv.ParseFloat(quant, 64); err == nil {
			b.MbS = f
			b.Measured |= MbS
		}
	case "B/op":
		if i, err := strconv.ParseUint(quant, 10, 64); err == nil {
			b.BOp = i
			b.Measured |= BOp
		}
	case "allocs/op":
		if i, err := strconv.ParseUint(quant, 10, 64); err == nil {
			b.AllocsOp = i
			b.Measured |= AllocsOp
		}
	}
}

func (b *Bench) String() string {
	buf := new(bytes.Buffer)
	fmt.Fprintf(buf, "%s %d", b.Name, b.N)
	if b.Measured&NsOp != 0 {
		fmt.Fprintf(buf, " %.2f ns/op", b.NsOp)
	}
	if b.Measured&MbS != 0 {
		fmt.Fprintf(buf, " %.2f MB/s", b.MbS)
	}
	if b.Measured&BOp != 0 {
		fmt.Fprintf(buf, " %d B/op", b.BOp)
	}
	if b.Measured&AllocsOp != 0 {
		fmt.Fprintf(buf, " %d allocs/op", b.AllocsOp)
	}
	return buf.String()
}

// BenchSet is a collection of benchmarks from one
// testing.B run, keyed by name to facilitate comparison.
type BenchSet map[string][]*Bench

// Parse extracts a BenchSet from testing.B output. Parse
// preserves the order of benchmarks that have identical names.
func ParseBenchSet(r io.Reader) (BenchSet, error) {
	bb := make(BenchSet)
	scan := bufio.NewScanner(r)
	ord := 0
	for scan.Scan() {
		if b, err := ParseLine(scan.Text()); err == nil {
			b.ord = ord
			ord++
			old := bb[b.Name]
			if *best && old != nil {
				if old[0].NsOp < b.NsOp {
					continue
				}
				b.ord = old[0].ord
				bb[b.Name] = old[:0]
			}
			bb[b.Name] = append(bb[b.Name], b)
		}
	}

	if err := scan.Err(); err != nil {
		return nil, err
	}

	return bb, nil
}
