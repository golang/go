// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestShapeNameLength tests that an unwieldy type argument doesn't leak into
// the names of every function instantiated with it. Shape names are embedded
// in those names, so without a bound a single large struct or an interface
// with many methods makes nm output, stack traces and pprof graphs unreadable.
// See go.dev/issue/65030 and go.dev/issue/71535.
func TestShapeNameLength(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	// Big and BigStruct are over the shape name limit and named, so their
	// shapes are named after them. The anonymous struct is over the limit
	// with no name to borrow. Small fits under the limit.
	const src = `package p

type Big interface {
	Eval(Context) Big
	Inner() Big
	ProgString() string
	Rank() int
	Sprint(*Config) string
	String() string
	shrink() Big
	toType(string, *Config, int) Big
}

type Context struct{ _ int }
type Config struct{ _ int }

type Small interface{ Len() int }

type Holder[T any] struct{ v []T }

func (h *Holder[T]) Append(x T) { h.v = append(h.v, x) }

type BigStruct struct {
	Field0, Field1, Field2, Field3, Field4 int
	Field5, Field6, Field7, Field8, Field9 int
}

// Wrapper[T] instantiated with a hashed shape gives a shape whose own
// tag contains a '#', which shapifying it again has to cope with.
type Wrapper[T any] struct {
	A, B, C, D, E T
	F, G, H, I, J T
	K, L, M, N, O T
}

func Nested[T any](x Wrapper[T]) {
	var h Holder[Wrapper[T]]
	h.Append(x)
}

var _ = Nested[Big]

var _ Holder[Big]
var _ Holder[Small]
var _ Holder[BigStruct]
var _ Holder[struct {
	Alpha, Bravo, Charlie, Delta, Echo   int
	Foxtrot, Golf, Hotel, India, Juliett int
}]
`

	dir := t.TempDir()
	srcGo := filepath.Join(dir, "p.go")
	if err := os.WriteFile(srcGo, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}
	obj := filepath.Join(dir, "p.o")

	gotool := testenv.GoToolPath(t)
	cmd := testenv.Command(t, gotool, "tool", "compile", "-p", "p", "-o", obj, srcGo)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, out)
	}

	cmd = testenv.Command(t, gotool, "tool", "nm", obj)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, out)
	}
	syms := string(out)

	// Big is over the limit, so its shape is named after it rather than
	// spelled out. Big can't even be written as a literal, as it refers to
	// itself, so its name is the only useful thing to call the shape.
	if !strings.Contains(syms, "go.shape.p.Big") {
		t.Errorf("no go.shape.p.Big symbol in nm output\n%s", syms)
	}
	if strings.Contains(syms, "go.shape.interface { Eval(") {
		t.Errorf("Big's full method set still appears in symbol names")
	}

	if !strings.Contains(syms, "go.shape.p.BigStruct") {
		t.Errorf("no go.shape.p.BigStruct symbol in nm output\n%s", syms)
	}

	// The shape name has to be qualified by package path: it is the shape's
	// identity, and an unqualified name would make two same-named types in
	// different packages share one shape.
	if regexp.MustCompile(`go\.shape\.Big\b`).MatchString(syms) {
		t.Errorf("Big's shape is named without its package\n%s", syms)
	}

	// An over-limit type argument with no name to borrow falls back to a
	// hash of the underlying type.
	if !regexp.MustCompile(`go\.shape\.[0-9a-f]{64}\b`).MatchString(syms) {
		t.Errorf("no go.shape.<hash> symbol in nm output\n%s", syms)
	}

	// The Small shape is under the limit, so it's still spelled out. Hashing
	// short shapes would lose readability for no benefit.
	if !strings.Contains(syms, "go.shape.interface { Len() int }") {
		t.Errorf("Small's shape was hashed; want it spelled out\n%s", syms)
	}

	// Shapifying a shape reproduces that shape's own name, so a generic
	// type instantiated with a named shape names its shape after the
	// instantiation, nesting and all.
	if !strings.Contains(syms, "go.shape.p.Wrapper[go.shape.p.Big]") {
		t.Errorf("no nested go.shape.p.Wrapper[go.shape.p.Big] symbol in nm output\n%s", syms)
	}

	// Shapify has to be idempotent, or the shaped object for an
	// instantiation is looked up under a different name than it was
	// defined with. That shows up as a duplicate symbol, so simply
	// compiling the package above covers it; assert the symbols are
	// distinct too, in case the duplicate check ever loosens.
	seen := map[string]bool{}
	for _, line := range strings.Split(syms, "\n") {
		f := strings.Fields(line)
		if len(f) < 3 || f[1] != "T" {
			continue
		}
		if name := f[2]; seen[name] {
			t.Errorf("duplicate text symbol %s", name)
		} else {
			seen[name] = true
		}
	}
}
