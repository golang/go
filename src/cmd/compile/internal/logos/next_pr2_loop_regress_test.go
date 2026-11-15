// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"
)

func TestNextLoopEmptyAndReceiverStayTasks(t *testing.T) {
	imp := newTestImporter(t)
	scan := func(t *testing.T, body string) (int, int) {
		t.Helper()
		call := "func d(x int) int { return x * 2 }\n"
		edits, tasks := scanSnippet(t, imp, "package p\n\n"+call+body+"\n")
		return len(edits), len(tasks)
	}
	t.Run("emptyComposite", func(t *testing.T) {
		body := "func g(xs []int) []int {\n\tys := []int{}\n\tfor i := 0; i < len(xs); i++ {\n\t\tys = append(ys, d(xs[i]))\n\t}\n\treturn ys\n}"
		if n, _ := scan(t, body); n != 0 {
			t.Fatalf("empty composite: got %d edits, want 0", n)
		}
	})
	t.Run("packageMutator", func(t *testing.T) {
		body := "var xs = []int{1, 2, 3}\nfunc shrink(x int) int { xs = xs[:1]; return x }\nfunc g() []int {\n\tvar ys []int\n\tfor i := 0; i < len(xs); i++ {\n\t\tys = append(ys, shrink(xs[i]))\n\t}\n\treturn ys\n}"
		if n, tasks := scan(t, body); n != 0 || tasks == 0 {
			t.Fatalf("package mutator: edits=%d tasks=%d, want 0 edits and a task", n, tasks)
		}
	})
	t.Run("localPure", func(t *testing.T) {
		body := "func g(xs []int) []int {\n\tvar ys []int\n\tfor i := 0; i < len(xs); i++ {\n\t\tys = append(ys, d(xs[i]))\n\t}\n\treturn ys\n}"
		edits, _ := scanSnippet(t, imp, "package p\n\nfunc d(x int) int { return x * 2 }\n"+body+"\n")
		if len(edits) != 1 {
			t.Fatalf("local pure: got %d edits, want 1", len(edits))
		}
	})
	t.Run("capturedClosure", func(t *testing.T) {
		body := "func g(xs []int) []int {\n\tvar ys []int\n\tf := func(x int) int { xs = xs[:1]; return x }\n\tfor i := 0; i < len(xs); i++ {\n\t\tys = append(ys, f(xs[i]))\n\t}\n\treturn ys\n}"
		if n, tasks := scan(t, body); n != 0 || tasks == 0 {
			t.Fatalf("captured closure: edits=%d tasks=%d, want 0 edits and a task", n, tasks)
		}
	})
}
