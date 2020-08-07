// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"io"
	"math/rand"
	"strconv"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

type progressTracker struct {
	client                   protocol.Client
	supportsWorkDoneProgress bool

	mu         sync.Mutex
	inProgress map[protocol.ProgressToken]*workDone
}

func newProgressTracker(client protocol.Client) *progressTracker {
	return &progressTracker{
		client:     client,
		inProgress: make(map[protocol.ProgressToken]*workDone),
	}
}

// start issues a $/progress notification to begin a unit of work on the
// server. The returned WorkDone handle may be used to report incremental
// progress, and to report work completion. In particular, it is an error to
// call start and not call end(...) on the returned WorkDone handle.
//
// If token is empty, a token will be randomly generated.
//
// The progress item is considered cancellable if the given cancel func is
// non-nil.
//
// Example:
//  func Generate(ctx) (err error) {
//    ctx, cancel := context.WithCancel(ctx)
//    defer cancel()
//    work := s.progress.start(ctx, "generate", "running go generate", cancel)
//    defer func() {
//      if err != nil {
//        work.end(ctx, fmt.Sprintf("generate failed: %v", err))
//      } else {
//        work.end(ctx, "done")
//      }
//    }()
//    // Do the work...
//  }
//
func (t *progressTracker) start(ctx context.Context, title, message string, token protocol.ProgressToken, cancel func()) *workDone {
	wd := &workDone{
		client: t.client,
		token:  token,
		cancel: cancel,
	}
	if !t.supportsWorkDoneProgress {
		wd.startErr = errors.New("workdone reporting is not supported")
		return wd
	}
	if wd.token == nil {
		wd.token = strconv.FormatInt(rand.Int63(), 10)
		err := wd.client.WorkDoneProgressCreate(ctx, &protocol.WorkDoneProgressCreateParams{
			Token: wd.token,
		})
		if err != nil {
			wd.startErr = err
			event.Error(ctx, "starting work for "+title, err)
			return wd
		}
	}
	t.mu.Lock()
	t.inProgress[wd.token] = wd
	t.mu.Unlock()
	wd.cleanup = func() {
		t.mu.Lock()
		delete(t.inProgress, token)
		t.mu.Unlock()
	}
	err := wd.client.Progress(ctx, &protocol.ProgressParams{
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

func (t *progressTracker) cancel(ctx context.Context, token protocol.ProgressToken) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	wd, ok := t.inProgress[token]
	if !ok {
		return errors.Errorf("token %q not found in progress", token)
	}
	if wd.cancel == nil {
		return errors.Errorf("work %q is not cancellable", token)
	}
	wd.cancel()
	return nil
}

// newProgressWriter returns an io.WriterCloser that can be used
// to report progress on a command based on the client capabilities.
func (t *progressTracker) newWriter(ctx context.Context, title, beginMsg, msg string, token protocol.ProgressToken, cancel func()) io.WriteCloser {
	if t.supportsWorkDoneProgress {
		wd := t.start(ctx, title, beginMsg, token, cancel)
		return &workDoneWriter{ctx, wd}
	}
	mw := &messageWriter{ctx, cancel, t.client}
	mw.start(msg)
	return mw
}

// workDone represents a unit of work that is reported to the client via the
// progress API.
type workDone struct {
	client   protocol.Client
	startErr error
	token    protocol.ProgressToken
	cancel   func()
	cleanup  func()
}

// report reports an update on WorkDone report back to the client.
func (wd *workDone) report(ctx context.Context, message string, percentage float64) error {
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

// end reports a workdone completion back to the client.
func (wd *workDone) end(ctx context.Context, message string) error {
	if wd.startErr != nil {
		return wd.startErr
	}
	err := wd.client.Progress(ctx, &protocol.ProgressParams{
		Token: wd.token,
		Value: &protocol.WorkDoneProgressEnd{
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
	wd  *workDone
}

func (wdw *workDoneWriter) Write(p []byte) (n int, err error) {
	return len(p), wdw.wd.report(wdw.ctx, string(p), 0)
}

func (wdw *workDoneWriter) Close() error {
	return wdw.wd.end(wdw.ctx, "finished")
}
