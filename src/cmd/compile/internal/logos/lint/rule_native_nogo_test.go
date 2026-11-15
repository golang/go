// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"
	"testing"

	"internal/testenv"
)

func TestLayoutConstant(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

type payload struct {
	a int32
	b int64
	c string
}

type head struct {
	first int8
	second int8
}
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"sizeof match", `
//logos:sizeof int64
const sizeInt64 = 8
`, nil},
		{"sizeof mismatch", `
//logos:sizeof int64
const wrongSize = 4
`, []string{"15 correctness::layout_constant_mismatch forbid"}},
		{"alignof match", `
//logos:alignof int64
const alignInt64 = 8
`, nil},
		{"offsetof match", `
//logos:offsetof head.second
const offSecond = 1
`, nil},
		{"offsetof mismatch", `
//logos:offsetof payload.b
const wrongOff = 4
`, []string{"15 correctness::layout_constant_mismatch forbid"}},
		{"sizeof struct match", `
//logos:sizeof payload
const sizePayload = 32
`, nil},
		{"unknown type", `
//logos:sizeof nosuchtype
const missing = 0
`, []string{"15 correctness::layout_constant_mismatch forbid"}},
		{"unknown field", `
//logos:sizeof payload.nofield
const missingField = 0
`, []string{"15 correctness::layout_constant_mismatch forbid"}},
		{"not a const", `
//logos:sizeof int64
var notConst = 8
`, []string{"15 correctness::layout_constant_mismatch forbid"}},
		{"misplaced directive", `
var x = 1

//logos:sizeof int64

func f() {
	_ = x
}
`, []string{"16 correctness::layout_constant_mismatch forbid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name+"/lgo", func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", head+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
		t.Run(tt.name+"/go", func(t *testing.T) {
			want := make([]string, 0, len(tt.want))
			for _, w := range tt.want {
				goW, _ := strings.CutSuffix(w, " forbid")
				want = append(want, goW+" deny")
			}
			got := summarize(Run(checkSrc(t, KindGo, "x.go", head+tt.body)))
			if !slices.Equal(got, want) {
				t.Errorf("results = %q, want %q", got, want)
			}
		})
	}
}

func TestAtomicPlainInt64(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import "sync/atomic"

var plain int64

var plainU uint64

var safe atomic.Int64

var small int32
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"load plain", `
func reads() int64 {
	return atomic.LoadInt64(&plain)
}
`, []string{"5 portability::atomic_plain_int64 warn"}},
		{"add plain uint64", `
func adds() {
	atomic.AddUint64(&plainU, 1)
}
`, []string{"7 portability::atomic_plain_int64 warn"}},
		{"cas plain", `
func swaps() {
	atomic.CompareAndSwapInt64(&plain, 0, 1)
}
`, []string{"5 portability::atomic_plain_int64 warn"}},
		{"typed wrapper", `
func reads() int64 {
	return safe.Load()
}
`, nil},
		{"32-bit ops", `
func reads() int32 {
	return atomic.LoadInt32(&small)
}
`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := checkSrc(t, KindGo, "x.go", head+tt.body)
			in.Policy = pedanticWarn()
			got := summarize(Run(in))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
	t.Run("reports under the default policy", func(t *testing.T) {
		in := checkSrc(t, KindGo, "x.go", head+`
func reads() int64 {
	return atomic.LoadInt64(&plain)
}
`)
		got := summarize(Run(in))
		if want := []string{"5 portability::atomic_plain_int64 warn"}; !slices.Equal(got, want) {
			t.Errorf("results = %q, want %q", got, want)
		}
	})
	t.Run("autofix names the wrapper type", func(t *testing.T) {
		for _, tt := range []struct {
			name string
			body string
			want string
		}{
			{"int64", `
func reads() int64 {
	return atomic.LoadInt64(&plain)
}
`, "atomic.Int64"},
			{"uint64", `
func adds() {
	atomic.AddUint64(&plainU, 1)
}
`, "atomic.Uint64"},
		} {
			t.Run(tt.name, func(t *testing.T) {
				in := checkSrc(t, KindGo, "x.go", head+tt.body)
				var fixes []Fix
				for _, r := range Run(in) {
					if r.Rule == "portability::atomic_plain_int64" {
						fixes = r.Fixes
					}
				}
				if len(fixes) != 1 || fixes[0].Replacement != tt.want {
					t.Errorf("fixes = %+v, want one fix with replacement %q", fixes, tt.want)
				}
			})
		}
	})
}

func TestLinknameMismatch(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import (
	"sync"
	_ "unsafe"
)
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"signature match", `
//go:linkname goodUnlock sync.(*Mutex).Unlock
func goodUnlock(m *sync.Mutex)
`, nil},
		{"signature mismatch", `
//go:linkname badLock sync.(*Mutex).Lock
func badLock(n int)

var keep = sync.Mutex{}
`, []string{"9 correctness::linkname_mismatch deny"}},
		{"method mismatch", `
//go:linkname badWait sync.WaitGroup.Done
func badWait(x int)

var keep = sync.Mutex{}
`, []string{"9 correctness::linkname_mismatch deny"}},
		{"detached directive", `
//go:linkname detached runtime.fastrand

var keep = sync.Mutex{}
`, nil},
		{"package not imported", `
//go:linkname notImported os.Getppid
func notImported()

var keep = sync.Mutex{}
`, nil},
		{"no remote name", `
//go:linkname localOnly
func localOnly()

var keep = sync.Mutex{}
`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindGo, "x.go", head+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLinknameMismatchTransitiveImport(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

import (
	"fmt"
	_ "unsafe"
)

//go:linkname badLock sync.(*Mutex).Lock
func badLock(n int)

var keep = fmt.Sprint
`
	got := summarize(Run(checkSrc(t, KindGo, "x.go", src)))
	want := []string{"9 correctness::linkname_mismatch deny"}
	if !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}

func TestUnsafeOutsideUnsafeFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

import "unsafe"

var slot unsafe.Pointer
`
	t.Run("plain file reports", func(t *testing.T) {
		in := checkSrc(t, KindGo, "x.go", src)
		in.Policy = pedanticWarn()
		got := summarize(Run(in))
		if want := []string{"3 pedantic::unsafe_outside_unsafe_file warn"}; !slices.Equal(got, want) {
			t.Errorf("results = %q, want %q", got, want)
		}
	})
	t.Run("unsafe file exempt", func(t *testing.T) {
		in := checkSrc(t, KindGo, "x_unsafe.go", src)
		in.Policy = pedanticWarn()
		got := summarize(Run(in))
		if len(got) != 0 {
			t.Errorf("results = %q, want none", got)
		}
	})
	t.Run("test file exempt", func(t *testing.T) {
		in := checkSrc(t, KindGo, "x_test.go", src)
		in.Policy = pedanticWarn()
		got := summarize(Run(in))
		if len(got) != 0 {
			t.Errorf("results = %q, want none", got)
		}
	})
	t.Run("lgo unsafe suffix exempt", func(t *testing.T) {
		in := checkSrc(t, KindLgo, "x_unsafe.lgo", src)
		in.Policy = pedanticWarn()
		got := summarize(Run(in))
		if len(got) != 0 {
			t.Errorf("results = %q, want none", got)
		}
	})
}
func TestArenaFallbackGo(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func f() int {
	return 1
}
`
	for _, tc := range []struct {
		kind Kind
		name string
	}{
		{KindGo, "x.go"},
		{KindLgo, "x.lgo"},
	} {
		in := checkSrc(t, tc.kind, tc.name, src)
		found := []FallbackFinding{{File: tc.name, Line: 4, Col: 2, Message: "arena.New[int] falls back to the heap"}}
		want := []string{"4 perf::arena_fallback warn"}
		if got := summarize(RunFallback(in, found)); !slices.Equal(got, want) {
			t.Errorf("kind %v: RunFallback = %q, want %q", tc.kind, got, want)
		}
	}
}
