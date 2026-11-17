// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"fmt"
	"go/build"
	"go/build/constraint"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

func variantTagTestContext(t *testing.T) {
	t.Helper()
	saved := cfg.BuildContext
	t.Cleanup(func() { cfg.BuildContext = saved })
	cfg.BuildContext = build.Default
	cfg.BuildContext.GOOS = "linux"
	cfg.BuildContext.GOARCH = "amd64"
	cfg.BuildContext.BuildTags = nil
}

func variantTagTestFile(t *testing.T, expr string) (string, string) {
	t.Helper()
	dir, name := t.TempDir(), "variant.lgo"
	source := "//go:build " + expr + "\n\npackage variants\n"
	if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0666); err != nil {
		t.Fatal(err)
	}
	return dir, name
}

func assertVariantTagMatch(t *testing.T, dir, name, goos, goarch string, tags []string) {
	t.Helper()
	ctxt := cfg.BuildContext
	ctxt.GOOS, ctxt.GOARCH = goos, goarch
	ctxt.BuildTags = tags
	ctxt.ToolTags = append(slices.Clone(ctxt.ToolTags), "logos")
	match, err := ctxt.MatchFile(dir, name)
	if err != nil || !match {
		t.Errorf("selected %s/%s tags=%q do not select source: match=%v, error=%v", goos, goarch, tags, match, err)
	}
}

func TestFindVariantContextTagCounts(t *testing.T) {
	variantTagTestContext(t)
	for _, count := range []int{strconv.IntSize - 1, strconv.IntSize, strconv.IntSize + 1, 128} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			want := make([]string, count)
			for i := range want {
				want[i] = fmt.Sprintf("variant%d", i)
			}
			dir, name := variantTagTestFile(t, strings.Join(want, " && "))
			goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, nil)
			if !ok {
				t.Fatalf("no context for a satisfiable %d-tag conjunction", count)
			}
			if goos != "linux" || goarch != "amd64" {
				t.Errorf("context = %s/%s, want linux/amd64", goos, goarch)
			}
			slices.Sort(want)
			if !slices.Equal(tags, want) {
				t.Errorf("tags = %q, want %q", tags, want)
			}
			assertVariantTagMatch(t, dir, name, goos, goarch, tags)
		})
	}
}

func TestFindVariantContextConstraintReadCount(t *testing.T) {
	variantTagTestContext(t)
	reads := 0
	cfg.BuildContext.OpenFile = func(path string) (io.ReadCloser, error) {
		reads++
		return os.Open(path)
	}
	dir, name := variantTagTestFile(t, "a && b && c && d && e && f && g && h && i && j && k && l")
	if _, _, _, ok := findVariantContextFromLinuxAMD64(dir, name, nil); !ok {
		t.Fatal("no context for a satisfiable conjunction")
	}
	if reads != 1 {
		t.Errorf("MatchFile read source %d times for one conjunction, want one final validation", reads)
	}
}

func TestFindVariantContextCanUnsetHostTags(t *testing.T) {
	variantTagTestContext(t)
	for _, expr := range []string{"!selected", "!selected && alternate"} {
		t.Run(expr, func(t *testing.T) {
			dir, name := variantTagTestFile(t, expr)
			host := []string{"selected", "untouched"}
			goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, host)
			if !ok {
				t.Fatal("no context after disabling a named custom tag")
			}
			if slices.Contains(tags, "selected") || !slices.Contains(tags, "untouched") {
				t.Errorf("tags = %q, want selected disabled and untouched retained", tags)
			}
			if !slices.Equal(host, []string{"selected", "untouched"}) {
				t.Errorf("input tags mutated: %q", host)
			}
			assertVariantTagMatch(t, dir, name, goos, goarch, tags)
		})
	}
}

func TestFindVariantContextConstraintOracle(t *testing.T) {
	variantTagTestContext(t)
	// The exhaustive oracle is deliberately independent of the selector.
	// Repeated variables force backtracking across an earlier disjunction.
	for _, expression := range []string{
		"a", "!a", "!(!a)", "a && !a", "a || !a",
		"(a || b) && !a", "(a || b) && (!a || c) && !b",
		"!((a && b) || (!a && c)) && (b || c)",
		"(a || b) && (!a || b) && (a || !b) && (!a || !b)",
		"a && (b || c) && !(a && b)",
	} {
		t.Run(expression, func(t *testing.T) {
			dir, name := variantTagTestFile(t, expression)
			expr, err := constraint.Parse("//go:build " + expression)
			if err != nil {
				t.Fatal(err)
			}
			want := false
			for mask := range 8 {
				if expr.Eval(func(tag string) bool { return mask&(1<<strings.Index("abc", tag)) != 0 }) {
					want = true
				}
			}
			goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, nil)
			if ok != want {
				t.Fatalf("found=%v, exhaustive oracle=%v", ok, want)
			}
			if ok {
				assertVariantTagMatch(t, dir, name, goos, goarch, tags)
			}
		})
	}
}

func TestFindVariantContextFixedTags(t *testing.T) {
	variantTagTestContext(t)
	for _, expression := range []string{"linux && windows", "cgo && !cgo", "!go1.1", "go1.999", "goexperiment.unavailable"} {
		t.Run(expression, func(t *testing.T) {
			dir, name := variantTagTestFile(t, expression)
			if goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, nil); ok {
				t.Errorf("found impossible context %s/%s tags=%q", goos, goarch, tags)
			}
		})
	}
}
