// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"math/rand"
	"strconv"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/xcontext"
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

// start notifies the client of work being done on the server. It uses either
// ShowMessage RPCs or $/progress messages, depending on the capabilities of
// the client.  The returned WorkDone handle may be used to report incremental
// progress, and to report work completion. In particular, it is an error to
// call start and not call end(...) on the returned WorkDone handle.
//
// If token is empty, a token will be randomly generated.
//
// The progress item is considered cancellable if the given cancel func is
// non-nil. In this case, cancel is called when the work done
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
		ctx:    xcontext.Detach(ctx),
		client: t.client,
		token:  token,
		cancel: cancel,
	}
	if !t.supportsWorkDoneProgress {
		// Previous iterations of this fallback attempted to retain cancellation
		// support by using ShowMessageCommand with a 'Cancel' button, but this is
		// not ideal as the 'Cancel' dialog stays open even after the command
		// completes.
		//
		// Just show a simple message. Clients can implement workDone progress
		// reporting to get cancellation support.
		if err := wd.client.ShowMessage(wd.ctx, &protocol.ShowMessageParams{
			Type:    protocol.Log,
			Message: message,
		}); err != nil {
			event.Error(ctx, "showing start message for "+title, err)
		}
		return wd
	}
	if wd.token == nil {
		token = strconv.FormatInt(rand.Int63(), 10)
		err := wd.client.WorkDoneProgressCreate(ctx, &protocol.WorkDoneProgressCreateParams{
			Token: token,
		})
		if err != nil {
			wd.err = err
			event.Error(ctx, "starting work for "+title, err)
			return wd
		}
		wd.token = token
	}
	// At this point we have a token that the client knows about. Store the token
	// before starting work.
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
	wd.doCancel()
	return nil
}

// workDone represents a unit of work that is reported to the client via the
// progress API.
type workDone struct {
	// ctx is detached, for sending $/progress updates.
	ctx    context.Context
	client protocol.Client
	// If token is nil, this workDone object uses the ShowMessage API, rather
	// than $/progress.
	token protocol.ProgressToken
	// err is set if progress reporting is broken for some reason (for example,
	// if there was an initial error creating a token).
	err error

	cancelMu  sync.Mutex
	cancelled bool
	cancel    func()

	cleanup func()
}

func (wd *workDone) doCancel() {
	wd.cancelMu.Lock()
	defer wd.cancelMu.Unlock()
	if !wd.cancelled {
		wd.cancel()
	}
}

// report reports an update on WorkDone report back to the client.
func (wd *workDone) report(message string, percentage float64) {
	wd.cancelMu.Lock()
	cancelled := wd.cancelled
	wd.cancelMu.Unlock()
	if cancelled {
		return
	}
	if wd.err != nil || wd.token == nil {
		// Not using the workDone API, so we do nothing. It would be far too spammy
		// to send incremental messages.
		return
	}
	err := wd.client.Progress(wd.ctx, &protocol.ProgressParams{
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
	if err != nil {
		event.Error(wd.ctx, "reporting progress", err)
	}
}

// end reports a workdone completion back to the client.
func (wd *workDone) end(message string) {
	var err error
	switch {
	case wd.err != nil:
		// There is a prior error.
	case wd.token == nil:
		// We're falling back to message-based reporting.
		err = wd.client.ShowMessage(wd.ctx, &protocol.ShowMessageParams{
			Type:    protocol.Info,
			Message: message,
		})
	default:
		err = wd.client.Progress(wd.ctx, &protocol.ProgressParams{
			Token: wd.token,
			Value: &protocol.WorkDoneProgressEnd{
				Kind:    "end",
				Message: message,
			},
		})
	}
	if err != nil {
		event.Error(wd.ctx, "ending work", err)
	}
	if wd.cleanup != nil {
		wd.cleanup()
	}
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

// workDoneWriter wraps a workDone handle to provide a Writer interface,
// so that workDone reporting can more easily be hooked into commands.
type workDoneWriter struct {
	wd *workDone
}

func (wdw workDoneWriter) Write(p []byte) (n int, err error) {
	wdw.wd.report(string(p), 0)
	// Don't fail just because of a failure to report progress.
	return len(p), nil
}
