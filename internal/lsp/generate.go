// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"io"
	"log"
	"math/rand"
	"strconv"

	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/event"
	errors "golang.org/x/xerrors"
)

func (s *Server) runGenerate(ctx context.Context, dir string, recursive bool) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	token := strconv.FormatInt(rand.Int63(), 10)
	s.inProgressMu.Lock()
	s.inProgress[token] = cancel
	s.inProgressMu.Unlock()
	defer s.clearInProgress(token)

	er := &eventWriter{ctx: ctx}
	wc := s.newProgressWriter(ctx, cancel, token)
	defer wc.Close()
	args := []string{"-x"}
	if recursive {
		args = append(args, "./...")
	}
	inv := &gocommand.Invocation{
		Verb:       "generate",
		Args:       args,
		Env:        s.session.Options().Env,
		WorkingDir: dir,
	}
	stderr := io.MultiWriter(er, wc)
	err := inv.RunPiped(ctx, er, stderr)
	if err != nil && !errors.Is(ctx.Err(), context.Canceled) {
		log.Printf("generate: command error: %v", err)
		s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Error,
			Message: "go generate exited with an error, check gopls logs",
		})
	}
}

// eventWriter writes every incoming []byte to
// event.Print with the operation=generate tag
// to distinguish its logs from others.
type eventWriter struct {
	ctx context.Context
}

func (ew *eventWriter) Write(p []byte) (n int, err error) {
	event.Print(ew.ctx, string(p), tag.Operation.Of("generate"))
	return len(p), nil
}

// newProgressWriter returns an io.WriterCloser that can be used
// to report progress on the "go generate" command based on the
// client capabilities.
func (s *Server) newProgressWriter(ctx context.Context, cancel func(), token string) io.WriteCloser {
	var wc interface {
		io.WriteCloser
		start()
	}
	if s.supportsWorkDoneProgress {
		wc = &workDoneWriter{ctx, token, s.client}
	} else {
		wc = &messageWriter{ctx, cancel, s.client}
	}
	wc.start()
	return wc
}

// messageWriter implements progressWriter
// and only tells the user that "go generate"
// has started through window/showMessage but does not
// report anything afterwards. This is because each
// log shows up as a separate window and therefore
// would be obnoxious to show every incoming line.
// Request cancellation happens synchronously through
// the ShowMessageRequest response.
type messageWriter struct {
	ctx    context.Context
	cancel func()
	client protocol.Client
}

func (lw *messageWriter) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func (lw *messageWriter) start() {
	go func() {
		msg, err := lw.client.ShowMessageRequest(lw.ctx, &protocol.ShowMessageRequestParams{
			Type:    protocol.Log,
			Message: "go generate has started, check logs for progress",
			Actions: []protocol.MessageActionItem{{
				Title: "Cancel",
			}},
		})
		if err != nil {
			event.Error(lw.ctx, "error sending initial generate msg", err)
			return
		}
		if msg != nil && msg.Title == "Cancel" {
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

// workDoneWriter implements progressWriter
// that will send $/progress notifications
// to the client. Request cancellations
// happens separately through the
// window/workDoneProgress/cancel request
// in which case the given context will be rendered
// done.
type workDoneWriter struct {
	ctx    context.Context
	token  string
	client protocol.Client
}

func (wdw *workDoneWriter) Write(p []byte) (n int, err error) {
	return len(p), wdw.client.Progress(wdw.ctx, &protocol.ProgressParams{
		Token: wdw.token,
		Value: &protocol.WorkDoneProgressReport{
			Kind:        "report",
			Cancellable: true,
			Message:     string(p),
		},
	})
}

func (wdw *workDoneWriter) start() {
	err := wdw.client.WorkDoneProgressCreate(wdw.ctx, &protocol.WorkDoneProgressCreateParams{
		Token: wdw.token,
	})
	if err != nil {
		event.Error(wdw.ctx, "generate progress create", err)
		return
	}
	err = wdw.client.Progress(wdw.ctx, &protocol.ProgressParams{
		Token: wdw.token,
		Value: &protocol.WorkDoneProgressBegin{
			Kind:        "begin",
			Cancellable: true,
			Message:     "running go generate",
			Title:       "generate",
		},
	})
	if err != nil {
		event.Error(wdw.ctx, "generate progress begin", err)
	}
}

func (wdw *workDoneWriter) Close() error {
	return wdw.client.Progress(wdw.ctx, &protocol.ProgressParams{
		Token: wdw.token,
		Value: protocol.WorkDoneProgressEnd{
			Kind:    "end",
			Message: "finished",
		},
	})
}
