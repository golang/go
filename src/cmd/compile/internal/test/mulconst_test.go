// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConstantMultiplies(t *testing.T) {
	testenv.MustHaveGoRun(t)

	signs := []string{"", "u"}
	widths := []int{8, 16, 32, 64}

	// Make test code.
	var code bytes.Buffer
	fmt.Fprintf(&code, "package main\n")
	for _, b := range widths {
		for _, s := range signs {
			fmt.Fprintf(&code, "type test_%s%d struct {\n", s, b)
			fmt.Fprintf(&code, "    m %sint%d\n", s, b)
			fmt.Fprintf(&code, "    f func(%sint%d)%sint%d\n", s, b, s, b)
			fmt.Fprintf(&code, "}\n")
			fmt.Fprintf(&code, "var test_%s%ds []test_%s%d\n", s, b, s, b)
		}
	}
	for _, b := range widths {
		for _, s := range signs {
			lo := -256
			hi := 256
			if b == 8 {
				lo = -128
				hi = 127
			}
			if s == "u" {
				lo = 0
			}
			for i := lo; i <= hi; i++ {
				name := fmt.Sprintf("f_%s%d_%d", s, b, i)
				name = strings.ReplaceAll(name, "-", "n")
				fmt.Fprintf(&code, "func %s(x %sint%d) %sint%d {\n", name, s, b, s, b)
				fmt.Fprintf(&code, "    return x*%d\n", i)
				fmt.Fprintf(&code, "}\n")
				fmt.Fprintf(&code, "func init() {\n")
				fmt.Fprintf(&code, "    test_%s%ds = append(test_%s%ds, test_%s%d{%d, %s})\n", s, b, s, b, s, b, i, name)
				fmt.Fprintf(&code, "}\n")
			}
		}
	}
	fmt.Fprintf(&code, "func main() {\n")
	for _, b := range widths {
		for _, s := range signs {
			lo := -256
			hi := 256
			if s == "u" {
				lo = 0
			}
			fmt.Fprintf(&code, "    for _, tst := range test_%s%ds {\n", s, b)
			fmt.Fprintf(&code, "        for x := %d; x <= %d; x++ {\n", lo, hi)
			fmt.Fprintf(&code, "            y := %sint%d(x)\n", s, b)
			fmt.Fprintf(&code, "            if tst.f(y) != y*tst.m {\n")
			fmt.Fprintf(&code, "                panic(tst.m)\n")
			fmt.Fprintf(&code, "            }\n")
			fmt.Fprintf(&code, "        }\n")
			fmt.Fprintf(&code, "    }\n")
		}
	}
	fmt.Fprintf(&code, "}\n")

	fmt.Printf("CODE:\n%s\n", string(code.Bytes()))

	// Make test file
	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	err := os.WriteFile(src, code.Bytes(), 0644)
	if err != nil {
		t.Fatalf("write file failed: %v", err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go run failed: %v\n%s", err, out)
	}
	if len(out) > 0 {
		t.Fatalf("got output when expecting none: %s\n", string(out))
	}
}

// Benchmark multiplication of an integer by various constants.
//
// The comment above each sub-benchmark provides an example of how the
// target multiplication operation might be implemented using shift
// (multiplication by a power of 2), addition and subtraction
// operations. It is platform-dependent whether these transformations
// are actually applied.

var (
	mulSinkI32 int32
	mulSinkI64 int64
	mulSinkU32 uint32
	mulSinkU64 uint64
)

func BenchmarkMulconstI32(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkI32 = x
	})
	// 5x = 4x + x
	b.Run("5", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkI32 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkI32 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkI32 = x
	})
	// -120x = 8x - 120x
	b.Run("-120", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= -120
		}
		mulSinkI32 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkI32 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func(b *testing.B) {
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkI32 = x
	})
}

func BenchmarkMulconstI64(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkI64 = x
	})
	// 5x = 4x + x
	b.Run("5", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkI64 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkI64 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkI64 = x
	})
	// -120x = 8x - 120x
	b.Run("-120", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= -120
		}
		mulSinkI64 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkI64 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func(b *testing.B) {
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkI64 = x
	})
}

func BenchmarkMulconstU32(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkU32 = x
	})
	// 5x = 4x + x
	b.Run("5", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkU32 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkU32 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkU32 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkU32 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func(b *testing.B) {
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkU32 = x
	})
}

func BenchmarkMulconstU64(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkU64 = x
	})
	// 5x = 4x + x
	b.Run("5", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkU64 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkU64 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkU64 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkU64 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func(b *testing.B) {
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkU64 = x
	})
}
