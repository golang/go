// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package progress

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

type fakeClient struct {
	protocol.Client

	token protocol.ProgressToken

	mu                                        sync.Mutex
	created, begun, reported, messages, ended int
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
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checkToken(params.Token)
	c.created++
	return nil
}

func (c *fakeClient) Progress(ctx context.Context, params *protocol.ProgressParams) error {
	c.mu.Lock()
	defer c.mu.Unlock()
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

func (c *fakeClient) ShowMessage(context.Context, *protocol.ShowMessageParams) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages++
	return nil
}

func setup(token protocol.ProgressToken) (context.Context, *Tracker, *fakeClient) {
	c := &fakeClient{}
	tracker := NewTracker(c)
	tracker.SetSupportsWorkDoneProgress(true)
	return context.Background(), tracker, c
}

func TestProgressTracker_Reporting(t *testing.T) {
	for _, test := range []struct {
		name                                            string
		supported                                       bool
		token                                           protocol.ProgressToken
		wantReported, wantCreated, wantBegun, wantEnded int
		wantMessages                                    int
	}{
		{
			name:         "unsupported",
			wantMessages: 2,
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
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			tracker.supportsWorkDoneProgress = test.supported
			work := tracker.Start(ctx, "work", "message", test.token, nil)
			client.mu.Lock()
			gotCreated, gotBegun := client.created, client.begun
			client.mu.Unlock()
			if gotCreated != test.wantCreated {
				t.Errorf("got %d created tokens, want %d", gotCreated, test.wantCreated)
			}
			if gotBegun != test.wantBegun {
				t.Errorf("got %d work begun, want %d", gotBegun, test.wantBegun)
			}
			// Ignore errors: this is just testing the reporting behavior.
			work.Report(ctx, "report", 50)
			client.mu.Lock()
			gotReported := client.reported
			client.mu.Unlock()
			if gotReported != test.wantReported {
				t.Errorf("got %d progress reports, want %d", gotReported, test.wantCreated)
			}
			work.End(ctx, "done")
			client.mu.Lock()
			gotEnded, gotMessages := client.ended, client.messages
			client.mu.Unlock()
			if gotEnded != test.wantEnded {
				t.Errorf("got %d ended reports, want %d", gotEnded, test.wantEnded)
			}
			if gotMessages != test.wantMessages {
				t.Errorf("got %d messages, want %d", gotMessages, test.wantMessages)
			}
		})
	}
}

func TestProgressTracker_Cancellation(t *testing.T) {
	for _, token := range []protocol.ProgressToken{nil, 1, "a"} {
		ctx, tracker, _ := setup(token)
		var canceled bool
		cancel := func() { canceled = true }
		work := tracker.Start(ctx, "work", "message", token, cancel)
		if err := tracker.Cancel(work.Token()); err != nil {
			t.Fatal(err)
		}
		if !canceled {
			t.Errorf("tracker.cancel(...): cancel not called")
		}
	}
}
