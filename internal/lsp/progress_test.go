// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

type fakeClient struct {
	protocol.Client

	token protocol.ProgressToken

	created, begun, reported, ended int
}

func (c *fakeClient) checkToken(token protocol.ProgressToken) {
	if token == nil {
		panic("nil token in progress message")
	}
	if c.token != nil && c.token != token {
		panic(fmt.Errorf("invalid token in progress message: got %v, want %v", token, c.token))
	}
}

func (c *fakeClient) WorkDoneProgressCreate(ctx context.Context, params *protocol.WorkDoneProgressCreateParams) error {
	c.checkToken(params.Token)
	c.created++
	return nil
}

func (c *fakeClient) Progress(ctx context.Context, params *protocol.ProgressParams) error {
	c.checkToken(params.Token)
	switch params.Value.(type) {
	case *protocol.WorkDoneProgressBegin:
		c.begun++
	case *protocol.WorkDoneProgressReport:
		c.reported++
	case *protocol.WorkDoneProgressEnd:
		c.ended++
	default:
		panic(fmt.Errorf("unknown progress value %T", params.Value))
	}
	return nil
}

func setup(token protocol.ProgressToken) (context.Context, *progressTracker, *fakeClient) {
	c := &fakeClient{}
	tracker := newProgressTracker(c)
	tracker.supportsWorkDoneProgress = true
	return context.Background(), tracker, c
}

func TestProgressTracker_Reporting(t *testing.T) {
	for _, test := range []struct {
		name                                            string
		supported                                       bool
		token                                           protocol.ProgressToken
		wantReported, wantCreated, wantBegun, wantEnded int
	}{
		{
			name: "unsupported",
		},
		{
			name:         "random token",
			supported:    true,
			wantCreated:  1,
			wantBegun:    1,
			wantReported: 1,
			wantEnded:    1,
		},
		{
			name:         "string token",
			supported:    true,
			token:        "token",
			wantBegun:    1,
			wantReported: 1,
			wantEnded:    1,
		},
		{
			name:         "numeric token",
			supported:    true,
			token:        1,
			wantReported: 1,
			wantBegun:    1,
			wantEnded:    1,
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			ctx, tracker, client := setup(test.token)
			tracker.supportsWorkDoneProgress = test.supported
			work := tracker.start(ctx, "work", "message", test.token, nil)
			if got := client.created; got != test.wantCreated {
				t.Errorf("got %d created tokens, want %d", got, test.wantCreated)
			}
			if got := client.begun; got != test.wantBegun {
				t.Errorf("got %d work begun, want %d", got, test.wantBegun)
			}
			// Ignore errors: this is just testing the reporting behavior.
			work.report(ctx, "report", 50)
			if got := client.reported; got != test.wantReported {
				t.Errorf("got %d progress reports, want %d", got, test.wantCreated)
			}
			work.end(ctx, "done")
			if got := client.ended; got != test.wantEnded {
				t.Errorf("got %d ended reports, want %d", got, test.wantEnded)
			}
		})
	}
}

func TestProgressTracker_Cancellation(t *testing.T) {
	for _, token := range []protocol.ProgressToken{nil, 1, "a"} {
		ctx, tracker, _ := setup(token)
		var cancelled bool
		cancel := func() { cancelled = true }
		work := tracker.start(ctx, "work", "message", token, cancel)
		if err := tracker.cancel(ctx, work.token); err != nil {
			t.Fatal(err)
		}
		if !cancelled {
			t.Errorf("tracker.cancel(...): cancel not called")
		}
	}
}
