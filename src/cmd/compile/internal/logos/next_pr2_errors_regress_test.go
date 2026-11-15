// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"
)

func TestNextShadowedZeroNamesStayTasks(t *testing.T) {
	imp := newTestImporter(t)
	call := "func op() error { return nil }\n"
	t.Run("universeFalse", func(t *testing.T) {
		body := call + "func g(v bool) (bool, error) {\n\tif err := op(); err != nil {\n\t\treturn false, err\n\t}\n\treturn v, nil\n}"
		edits, tasks := scanSnippet(t, imp, "package p\n\n"+body+"\n")
		if len(edits) != 1 || len(tasks) != 0 {
			t.Fatalf("universe false: edits=%d tasks=%d, want 1, 0", len(edits), len(tasks))
		}
	})
	t.Run("universeNil", func(t *testing.T) {
		body := call + "func h(v *int) (*int, error) {\n\tif err := op(); err != nil {\n\t\treturn nil, err\n\t}\n\treturn v, nil\n}"
		edits, tasks := scanSnippet(t, imp, "package p\n\n"+body+"\n")
		if len(edits) != 1 || len(tasks) != 0 {
			t.Fatalf("universe nil: edits=%d tasks=%d, want 1, 0", len(edits), len(tasks))
		}
	})
	t.Run("shadowedFalse", func(t *testing.T) {
		body := call + "func g(false bool) (bool, error) {\n\tif err := op(); err != nil {\n\t\treturn false, err\n\t}\n\treturn false, nil\n}"
		edits, tasks := scanSnippet(t, imp, "package p\n\n"+body+"\n")
		if len(edits) != 0 {
			t.Fatalf("shadowed false: got %d edits %v, want 0", len(edits), edits)
		}
		if len(tasks) == 0 {
			t.Fatal("shadowed false: got 0 tasks, want a manual task")
		}
	})
	t.Run("shadowedNil", func(t *testing.T) {
		body := call + "func g(nil error) (error, error) {\n\tif err := op(); err != nil {\n\t\treturn nil, err\n\t}\n\treturn nil, nil\n}"
		edits, tasks := scanSnippet(t, imp, "package p\n\n"+body+"\n")
		if len(edits) != 0 {
			t.Fatalf("shadowed nil: got %d edits %v, want 0", len(edits), edits)
		}
		if len(tasks) == 0 {
			t.Fatal("shadowed nil: got 0 tasks, want a manual task")
		}
	})
}
