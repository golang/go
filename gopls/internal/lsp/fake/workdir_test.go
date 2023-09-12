// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

const sharedData = `
-- go.mod --
go 1.12
-- nested/README.md --
Hello World!
`

// newWorkdir sets up a temporary Workdir with the given txtar-encoded content.
// It also configures an eventBuffer to receive file event notifications. These
// notifications are sent synchronously for each operation, such that once a
// workdir file operation has returned the caller can expect that any relevant
// file notifications are present in the buffer.
//
// It is the caller's responsibility to call the returned cleanup function.
func newWorkdir(t *testing.T, txt string) (*Workdir, *eventBuffer, func()) {
	t.Helper()

	tmpdir, err := os.MkdirTemp("", "goplstest-workdir-")
	if err != nil {
		t.Fatal(err)
	}
	wd, err := NewWorkdir(tmpdir, UnpackTxt(txt))
	if err != nil {
		t.Fatal(err)
	}
	cleanup := func() {
		if err := os.RemoveAll(tmpdir); err != nil {
			t.Error(err)
		}
	}

	buf := new(eventBuffer)
	wd.AddWatcher(buf.onEvents)
	return wd, buf, cleanup
}

// eventBuffer collects events from a file watcher.
type eventBuffer struct {
	mu     sync.Mutex
	events []protocol.FileEvent
}

// onEvents collects adds events to the buffer; to be used with Workdir.AddWatcher.
func (c *eventBuffer) onEvents(_ context.Context, events []protocol.FileEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.events = append(c.events, events...)
}

// take empties the buffer, returning its previous contents.
func (c *eventBuffer) take() []protocol.FileEvent {
	c.mu.Lock()
	defer c.mu.Unlock()

	evts := c.events
	c.events = nil
	return evts
}

func TestWorkdir_ReadFile(t *testing.T) {
	wd, _, cleanup := newWorkdir(t, sharedData)
	defer cleanup()

	got, err := wd.ReadFile("nested/README.md")
	if err != nil {
		t.Fatal(err)
	}
	want := "Hello World!\n"
	if got := string(got); got != want {
		t.Errorf("reading workdir file, got %q, want %q", got, want)
	}
}

func TestWorkdir_WriteFile(t *testing.T) {
	wd, events, cleanup := newWorkdir(t, sharedData)
	defer cleanup()
	ctx := context.Background()

	tests := []struct {
		path     string
		wantType protocol.FileChangeType
	}{
		{"data.txt", protocol.Created},
		{"nested/README.md", protocol.Changed},
	}

	for _, test := range tests {
		if err := wd.WriteFile(ctx, test.path, "42"); err != nil {
			t.Fatal(err)
		}
		es := events.take()
		if got := len(es); got != 1 {
			t.Fatalf("len(events) = %d, want 1", got)
		}
		path := wd.URIToPath(es[0].URI)
		if path != test.path {
			t.Errorf("event path = %q, want %q", path, test.path)
		}
		if es[0].Type != test.wantType {
			t.Errorf("event type = %v, want %v", es[0].Type, test.wantType)
		}
		got, err := wd.ReadFile(test.path)
		if err != nil {
			t.Fatal(err)
		}
		want := "42"
		if got := string(got); got != want {
			t.Errorf("ws.ReadFile(%q) = %q, want %q", test.path, got, want)
		}
	}
}

// Test for file notifications following file operations.
func TestWorkdir_FileWatching(t *testing.T) {
	wd, events, cleanup := newWorkdir(t, "")
	defer cleanup()
	ctx := context.Background()

	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}

	type changeMap map[string]protocol.FileChangeType
	checkEvent := func(wantChanges changeMap) {
		gotChanges := make(changeMap)
		for _, e := range events.take() {
			gotChanges[wd.URIToPath(e.URI)] = e.Type
		}
		if diff := cmp.Diff(wantChanges, gotChanges); diff != "" {
			t.Errorf("mismatching file events (-want +got):\n%s", diff)
		}
	}

	must(wd.WriteFile(ctx, "foo.go", "package foo"))
	checkEvent(changeMap{"foo.go": protocol.Created})

	must(wd.RenameFile(ctx, "foo.go", "bar.go"))
	checkEvent(changeMap{"foo.go": protocol.Deleted, "bar.go": protocol.Created})

	must(wd.RemoveFile(ctx, "bar.go"))
	checkEvent(changeMap{"bar.go": protocol.Deleted})
}

func TestWorkdir_CheckForFileChanges(t *testing.T) {
	t.Skip("broken on darwin-amd64-10_12")
	wd, events, cleanup := newWorkdir(t, sharedData)
	defer cleanup()
	ctx := context.Background()

	checkChange := func(wantPath string, wantType protocol.FileChangeType) {
		if err := wd.CheckForFileChanges(ctx); err != nil {
			t.Fatal(err)
		}
		ev := events.take()
		if len(ev) == 0 {
			t.Fatal("no file events received")
		}
		gotEvt := ev[0]
		gotPath := wd.URIToPath(gotEvt.URI)
		// Only check relative path and Type
		if gotPath != wantPath || gotEvt.Type != wantType {
			t.Errorf("file events: got %v, want {Path: %s, Type: %v}", gotEvt, wantPath, wantType)
		}
	}
	// Sleep some positive amount of time to ensure a distinct mtime.
	if err := writeFileData("go.mod", []byte("module foo.test\n"), wd.RelativeTo); err != nil {
		t.Fatal(err)
	}
	checkChange("go.mod", protocol.Changed)
	if err := writeFileData("newFile", []byte("something"), wd.RelativeTo); err != nil {
		t.Fatal(err)
	}
	checkChange("newFile", protocol.Created)
	fp := wd.AbsPath("newFile")
	if err := os.Remove(fp); err != nil {
		t.Fatal(err)
	}
	checkChange("newFile", protocol.Deleted)
}

func TestSplitModuleVersionPath(t *testing.T) {
	tests := []struct {
		path                                string
		wantModule, wantVersion, wantSuffix string
	}{
		{"foo.com@v1.2.3/bar", "foo.com", "v1.2.3", "bar"},
		{"foo.com/module@v1.2.3/bar", "foo.com/module", "v1.2.3", "bar"},
		{"foo.com@v1.2.3", "foo.com", "v1.2.3", ""},
		{"std@v1.14.0", "std", "v1.14.0", ""},
		{"another/module/path", "another/module/path", "", ""},
	}

	for _, test := range tests {
		module, version, suffix := splitModuleVersionPath(test.path)
		if module != test.wantModule || version != test.wantVersion || suffix != test.wantSuffix {
			t.Errorf("splitModuleVersionPath(%q) =\n\t(%q, %q, %q)\nwant\n\t(%q, %q, %q)",
				test.path, module, version, suffix, test.wantModule, test.wantVersion, test.wantSuffix)
		}
	}
}
