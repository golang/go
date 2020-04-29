// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"io/ioutil"
	"os"
	"sort"
	"testing"
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
)

const data = `
-- go.mod --
go 1.12
-- nested/README.md --
Hello World!
`

func newWorkdir(t *testing.T) (*Workdir, <-chan []FileEvent, func()) {
	t.Helper()

	tmpdir, err := ioutil.TempDir("", "goplstest-workdir-")
	if err != nil {
		t.Fatal(err)
	}
	wd, err := NewWorkdir(tmpdir, data)
	if err != nil {
		t.Fatal(err)
	}
	cleanup := func() {
		if err := os.RemoveAll(tmpdir); err != nil {
			t.Error(err)
		}
	}

	fileEvents := make(chan []FileEvent)
	watch := func(_ context.Context, events []FileEvent) {
		fileEvents <- events
	}
	wd.AddWatcher(watch)
	return wd, fileEvents, cleanup
}

func TestWorkdir_ReadFile(t *testing.T) {
	wd, _, cleanup := newWorkdir(t)
	defer cleanup()

	got, err := wd.ReadFile("nested/README.md")
	if err != nil {
		t.Fatal(err)
	}
	want := "Hello World!\n"
	if got != want {
		t.Errorf("reading workdir file, got %q, want %q", got, want)
	}
}

func TestWorkdir_WriteFile(t *testing.T) {
	wd, events, cleanup := newWorkdir(t)
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
		es := <-events
		if got := len(es); got != 1 {
			t.Fatalf("len(events) = %d, want 1", got)
		}
		if es[0].Path != test.path {
			t.Errorf("event.Path = %q, want %q", es[0].Path, test.path)
		}
		if es[0].ProtocolEvent.Type != test.wantType {
			t.Errorf("event type = %v, want %v", es[0].ProtocolEvent.Type, test.wantType)
		}
		got, err := wd.ReadFile(test.path)
		if err != nil {
			t.Fatal(err)
		}
		want := "42"
		if got != want {
			t.Errorf("ws.ReadFile(%q) = %q, want %q", test.path, got, want)
		}
	}
}

func TestWorkdir_ListFiles(t *testing.T) {
	wd, _, cleanup := newWorkdir(t)
	defer cleanup()

	checkFiles := func(dir string, want []string) {
		files, err := wd.ListFiles(dir)
		if err != nil {
			t.Fatal(err)
		}
		sort.Strings(want)
		var got []string
		for p := range files {
			got = append(got, p)
		}
		sort.Strings(got)
		if len(got) != len(want) {
			t.Fatalf("ListFiles(): len = %d, want %d; got=%v; want=%v", len(got), len(want), got, want)
		}
		for i, f := range got {
			if f != want[i] {
				t.Errorf("ListFiles()[%d] = %s, want %s", i, f, want[i])
			}
		}
	}

	checkFiles(".", []string{"go.mod", "nested/README.md"})
	checkFiles("nested", []string{"nested/README.md"})
}

func TestWorkdir_CheckForFileChanges(t *testing.T) {
	t.Skip("broken on darwin-amd64-10_12")
	wd, events, cleanup := newWorkdir(t)
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	checkChange := func(path string, typ protocol.FileChangeType) {
		if err := wd.CheckForFileChanges(ctx); err != nil {
			t.Fatal(err)
		}
		var gotEvt FileEvent
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case ev := <-events:
			gotEvt = ev[0]
		}
		// Only check relative path and Type
		if gotEvt.Path != path || gotEvt.ProtocolEvent.Type != typ {
			t.Errorf("file events: got %v, want {Path: %s, Type: %v}", gotEvt, path, typ)
		}
	}
	// Sleep some positive amount of time to ensure a distinct mtime.
	time.Sleep(100 * time.Millisecond)
	if err := wd.writeFileData("go.mod", "module foo.test\n"); err != nil {
		t.Fatal(err)
	}
	checkChange("go.mod", protocol.Changed)
	if err := wd.writeFileData("newFile", "something"); err != nil {
		t.Fatal(err)
	}
	checkChange("newFile", protocol.Created)
	fp := wd.filePath("newFile")
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
