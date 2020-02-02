// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

const data = `
-- go.mod --
go 1.12
-- nested/README.md --
Hello World!
`

func newWorkspace(t *testing.T) (*Workspace, <-chan []FileEvent, func()) {
	t.Helper()

	ws, err := NewWorkspace("default", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	cleanup := func() {
		if err := ws.Close(); err != nil {
			t.Fatal(err)
		}
	}

	fileEvents := make(chan []FileEvent)
	watch := func(_ context.Context, events []FileEvent) {
		fileEvents <- events
	}
	ws.AddWatcher(watch)
	return ws, fileEvents, cleanup
}

func TestWorkspace_ReadFile(t *testing.T) {
	ws, _, cleanup := newWorkspace(t)
	defer cleanup()

	got, err := ws.ReadFile("nested/README.md")
	if err != nil {
		t.Fatal(err)
	}
	want := "Hello World!\n"
	if got != want {
		t.Errorf("reading workspace file, got %q, want %q", got, want)
	}
}

func TestWorkspace_WriteFile(t *testing.T) {
	ws, events, cleanup := newWorkspace(t)
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
		if err := ws.WriteFile(ctx, test.path, "42"); err != nil {
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
		got, err := ws.ReadFile(test.path)
		if err != nil {
			t.Fatal(err)
		}
		want := "42"
		if got != want {
			t.Errorf("ws.ReadFile(%q) = %q, want %q", test.path, got, want)
		}
	}
}
