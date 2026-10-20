// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"slices"
	"strings"
	"testing"
	"time"

	"cmd/compile/internal/syntax"
)

// contextBenchSrc builds a 200-type package with a method-rich target.
func contextBenchSrc() string {
	var b strings.Builder
	b.WriteString("package bench\n")
	for i := range 200 {
		b.WriteString("type T")
		writeBenchNum(&b, i)
		b.WriteString(" struct {\n\tNext *T")
		writeBenchNum(&b, (i+1)%200)
		b.WriteString("\n}\n")
	}
	b.WriteString("type Target struct {\n\tA T001\n\tB T002\n}\n")
	for i := range 5 {
		b.WriteString("func (t *Target) M")
		writeBenchNum(&b, i)
		b.WriteString("(a T003, b T004) (T005, error) {\n\treturn T005{}, nil\n}\n")
	}
	return b.String()
}

// writeBenchNum appends i as a three-digit suffix.
func writeBenchNum(b *strings.Builder, i int) {
	b.WriteByte(byte('0' + i/100))
	b.WriteByte(byte('0' + (i/10)%10))
	b.WriteByte(byte('0' + i%10))
}

// BenchmarkContext reports extraction latency over 10 counts.
func BenchmarkContext(b *testing.B) {
	src := contextBenchSrc()
	const counts = 10
	durs := make([]time.Duration, 0, counts)
	for range counts {
		start := time.Now()
		base := syntax.NewFileBase("bench.lgo")
		file, err := syntax.Parse(base, strings.NewReader(src), func(err error) { b.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
		if err != nil {
			b.Fatalf("parse: %v", err)
		}
		lines, err := ContextLines([]*syntax.File{file}, "Target")
		if err != nil {
			b.Fatalf("ContextLines: %v", err)
		}
		if len(lines) == 0 {
			b.Fatal("empty context")
		}
		durs = append(durs, time.Since(start))
	}
	slices.Sort(durs)
	b.Logf("p50=%v p95=%v p99=%v max=%v over %d counts (target 50ms)", durs[counts/2], durs[counts-1], durs[counts-1], durs[counts-1], counts)
}
