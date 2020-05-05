// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"io"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/xerrors"
)

// GenerateWorkDoneTitle is the title used in progress reporting for go
// generate commands. It is exported for testing purposes.
const GenerateWorkDoneTitle = "generate"

func (s *Server) runGenerate(ctx context.Context, dir string, recursive bool) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	er := &eventWriter{ctx: ctx, operation: "generate"}
	wc := s.newProgressWriter(ctx, GenerateWorkDoneTitle, "running go generate", cancel)
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
	if err != nil {
		event.Error(ctx, "generate: command error", err, tag.Directory.Of(dir))
		if !xerrors.Is(err, context.Canceled) {
			s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: "go generate exited with an error, check gopls logs",
			})
		}
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

// newProgressWriter returns an io.WriterCloser that can be used
// to report progress on a command based on the client capabilities.
func (s *Server) newProgressWriter(ctx context.Context, title, message string, cancel func()) io.WriteCloser {
	if s.supportsWorkDoneProgress {
		wd := s.StartWork(ctx, title, message, cancel)
		return &workDoneWriter{ctx, wd}
	}
	mw := &messageWriter{ctx, cancel, s.client}
	mw.start()
	return mw
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
