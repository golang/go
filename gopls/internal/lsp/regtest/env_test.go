// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"encoding/json"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func TestProgressUpdating(t *testing.T) {
	a := &Awaiter{
		state: State{
			outstandingWork: make(map[protocol.ProgressToken]*workProgress),
			startedWork:     make(map[string]uint64),
			completedWork:   make(map[string]uint64),
		},
	}
	ctx := context.Background()
	if err := a.onWorkDoneProgressCreate(ctx, &protocol.WorkDoneProgressCreateParams{
		Token: "foo",
	}); err != nil {
		t.Fatal(err)
	}
	if err := a.onWorkDoneProgressCreate(ctx, &protocol.WorkDoneProgressCreateParams{
		Token: "bar",
	}); err != nil {
		t.Fatal(err)
	}
	updates := []struct {
		token string
		value interface{}
	}{
		{"foo", protocol.WorkDoneProgressBegin{Kind: "begin", Title: "foo work"}},
		{"bar", protocol.WorkDoneProgressBegin{Kind: "begin", Title: "bar work"}},
		{"foo", protocol.WorkDoneProgressEnd{Kind: "end"}},
		{"bar", protocol.WorkDoneProgressReport{Kind: "report", Percentage: 42}},
	}
	for _, update := range updates {
		params := &protocol.ProgressParams{
			Token: update.token,
			Value: update.value,
		}
		data, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		var unmarshaled protocol.ProgressParams
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatal(err)
		}
		if err := a.onProgress(ctx, &unmarshaled); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := a.state.outstandingWork["foo"]; ok {
		t.Error("got work entry for \"foo\", want none")
	}
	got := *a.state.outstandingWork["bar"]
	want := workProgress{title: "bar work", percent: 42}
	if got != want {
		t.Errorf("work progress for \"bar\": %v, want %v", got, want)
	}
}
