// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	jsonv2 "encoding/json/v2"
)

func TestProxyURISpellingsShareOneDocument(t *testing.T) {
	f := newFixture(t, map[string]string{"a/a.lgo": aLgo}, fixtureOptions{})
	aPath := f.path("a/a.lgo")
	canonical := f.uri("a/a.lgo")
	localhostURI := "file://localhost" + strings.TrimPrefix(canonical, "file://")
	encodedURI := strings.Replace(canonical, "/a.lgo", "/%61.lgo", 1)
	upperURI := "file://LOCALHOST" + strings.TrimPrefix(canonical, "file://")
	f.open("a/a.lgo", 1)
	edited2 := aLgo + "// v2\n"
	edited3 := aLgo + "// v2\n// v3\n"
	edited4 := aLgo + "// v2\n// v3\n// v4\n"
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": localhostURI, "version": 2},
		"contentChanges": []any{map[string]any{"text": edited2}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": encodedURI, "version": 3},
		"contentChanges": []any{map[string]any{"text": edited3}},
	})
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": upperURI, "version": 4},
		"contentChanges": []any{map[string]any{"text": edited4}},
	})
	operand := shift(findPos(t, edited4, "ReadFile(path)?", 0), len("ReadFile("))
	f.client.call("textDocument/hover", positionParams(canonical, operand))
	calls := f.comp.lowerCalls()
	if len(calls) != 1 {
		t.Fatalf("three URI spellings compiled %d times, want the one shared run", len(calls))
	}
	buffers := f.comp.bufferCalls()
	last := buffers[len(buffers)-1]
	if len(last) != 1 {
		t.Fatalf("compile buffers hold %d documents, want the one shared document", len(last))
	}
	if string(last[aPath]) != edited4 {
		t.Errorf("hover text through the canonical spelling missed the version 4 edit through other spellings")
	}
	var got snapshotCountsResult
	if err := jsonv2.Unmarshal(f.client.call("logos/snapshotCounts", map[string]any{}), &got); err != nil {
		t.Fatal(err)
	}
	if want := map[string]int{filepath.Dir(aPath): 1}; !reflect.DeepEqual(got.Dirs, want) || got.DiagCache != 0 {
		t.Fatalf("snapshot counts = %+v, want dirs %v with an empty diagnostic cache", got, want)
	}
}
