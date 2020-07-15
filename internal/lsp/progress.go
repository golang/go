// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"strconv"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
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

// eventWriter writes every incoming []byte to
// event.Print with the operation=generate tag
// to distinguish its logs from others.
type eventWriter struct {
	ctx       context.Context
	operation string
}

func (ew *eventWriter) Write(p []byte) (n int, err error) {
	event.Log(ew.ctx, string(p), tag.Operation.Of(ew.operation))
	return len(p), nil
}

// newProgressWriter returns an io.WriterCloser that can be used
// to report progress on a command based on the client capabilities.
func (s *Server) newProgressWriter(ctx context.Context, title, beginMsg, msg string, cancel func()) io.WriteCloser {
	if s.supportsWorkDoneProgress {
		wd := s.StartWork(ctx, title, beginMsg, cancel)
		return &workDoneWriter{ctx, wd}
	}
	mw := &messageWriter{ctx, cancel, s.client}
	mw.start(msg)
	return mw
}

// messageWriter implements progressWriter and only tells the user that
// a command has started through window/showMessage, but does not report
// anything afterwards. This is because each log shows up as a separate window
// and therefore would be obnoxious to show every incoming line. Request
// cancellation happens synchronously through the ShowMessageRequest response.
type messageWriter struct {
	ctx    context.Context
	cancel func()
	client protocol.Client
}

func (lw *messageWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func (lw *messageWriter) start(msg string) {
	go func() {
		const cancel = "Cancel"
		item, err := lw.client.ShowMessageRequest(lw.ctx, &protocol.ShowMessageRequestParams{
			Type:    protocol.Log,
			Message: msg,
			Actions: []protocol.MessageActionItem{{
				Title: "Cancel",
			}},
		})
		if err != nil {
			event.Error(lw.ctx, "error sending message request", err)
			return
		}
		if item != nil && item.Title == "Cancel" {
			lw.cancel()
		}
	}()
}

func (lw *messageWriter) Close() error {
	return lw.client.ShowMessage(lw.ctx, &protocol.ShowMessageParams{
		Type:    protocol.Info,
		Message: "go generate has finished",
	})
}

// workDoneWriter implements progressWriter by sending $/progress notifications
// to the client. Request cancellations happens separately through the
// window/workDoneProgress/cancel request, in which case the given context will
// be rendered done.
type workDoneWriter struct {
	ctx context.Context
	wd  *WorkDone
}

func (wdw *workDoneWriter) Write(p []byte) (n int, err error) {
	return len(p), wdw.wd.Progress(wdw.ctx, string(p), 0)
}

func (wdw *workDoneWriter) Close() error {
	return wdw.wd.End(wdw.ctx, "finished")
}
