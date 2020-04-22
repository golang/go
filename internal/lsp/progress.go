// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"errors"
	"math/rand"
	"strconv"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
)

// WorkDone represents a unit of work that is reported to the client via the
// progress API.
type WorkDone struct {
	client   protocol.Client
	startErr error
	token    string
	cancel   func()
	cleanup  func()
}

// StartWork creates a unique token and issues a $/progress notification to
// begin a unit of work on the server. The returned WorkDone handle may be used
// to report incremental progress, and to report work completion. In
// particular, it is an error to call StartWork and not call End(...) on the
// returned WorkDone handle.
//
// The progress item is considered cancellable if the given cancel func is
// non-nil.
//
// Example:
//  func Generate(ctx) (err error) {
//    ctx, cancel := context.WithCancel(ctx)
//    defer cancel()
//    work := s.StartWork(ctx, "generate", "running go generate", cancel)
//    defer func() {
//      if err != nil {
//        work.End(ctx, fmt.Sprintf("generate failed: %v", err))
//      } else {
//        work.End(ctx, "done")
//      }
//    }()
//    // Do the work...
//  }
//
func (s *Server) StartWork(ctx context.Context, title, message string, cancel func()) *WorkDone {
	wd := &WorkDone{
		client: s.client,
		token:  strconv.FormatInt(rand.Int63(), 10),
		cancel: cancel,
	}
	if !s.supportsWorkDoneProgress {
		wd.startErr = errors.New("workdone reporting is not supported")
		return wd
	}
	err := wd.client.WorkDoneProgressCreate(ctx, &protocol.WorkDoneProgressCreateParams{
		Token: wd.token,
	})
	if err != nil {
		wd.startErr = err
		event.Error(ctx, "starting work for "+title, err)
		return wd
	}
	s.addInProgress(wd)
	wd.cleanup = func() {
		s.removeInProgress(wd.token)
	}
	err = wd.client.Progress(ctx, &protocol.ProgressParams{
		Token: wd.token,
		Value: &protocol.WorkDoneProgressBegin{
			Kind:        "begin",
			Cancellable: wd.cancel != nil,
			Message:     message,
			Title:       title,
		},
	})
	if err != nil {
		event.Error(ctx, "generate progress begin", err)
	}
	return wd
}

// Progress reports an update on WorkDone progress back to the client.
func (wd *WorkDone) Progress(ctx context.Context, message string, percentage float64) error {
	if wd.startErr != nil {
		return wd.startErr
	}
	return wd.client.Progress(ctx, &protocol.ProgressParams{
		Token: wd.token,
		Value: &protocol.WorkDoneProgressReport{
			Kind: "report",
			// Note that in the LSP spec, the value of Cancellable may be changed to
			// control whether the cancel button in the UI is enabled. Since we don't
			// yet use this feature, the value is kept constant here.
			Cancellable: wd.cancel != nil,
			Message:     message,
			Percentage:  percentage,
		},
	})
}

// End reports a workdone completion back to the client.
func (wd *WorkDone) End(ctx context.Context, message string) error {
	if wd.startErr != nil {
		return wd.startErr
	}
	err := wd.client.Progress(ctx, &protocol.ProgressParams{
		Token: wd.token,
		Value: protocol.WorkDoneProgressEnd{
			Kind:    "end",
			Message: message,
		},
	})
	if wd.cleanup != nil {
		wd.cleanup()
	}
	return err
}
