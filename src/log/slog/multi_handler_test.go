// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"
)

// mockFailingHandler is a handler that always returns an error
// from its Handle method.
type mockFailingHandler struct {
	Handler
	err error
}

func (h *mockFailingHandler) Handle(ctx context.Context, r Record) error {
	_ = h.Handler.Handle(ctx, r)
	return h.err
}

func TestMultiHandler(t *testing.T) {
	t.Run("Handle sends log to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, nil)
		h2 := NewJSONHandler(&buf2, nil)

		multi := NewMultiHandler(h1, h2)
		logger := New(multi)

		logger.Info("hello world", "user", "test")

		checkLogOutput(t, buf1.String(), "time="+textTimeRE+` level=INFO msg="hello world" user=test`)
		checkLogOutput(t, buf2.String(), `{"time":"`+jsonTimeRE+`","level":"INFO","msg":"hello world","user":"test"}`)
	})

	t.Run("Enabled returns true if any handler is enabled", func(t *testing.T) {
		h1 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelError})
		h2 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelInfo})

		multi := NewMultiHandler(h1, h2)

		if !multi.Enabled(context.Background(), LevelInfo) {
			t.Error("Enabled should be true for INFO level, but got false")
		}
		if !multi.Enabled(context.Background(), LevelError) {
			t.Error("Enabled should be true for ERROR level, but got false")
		}
	})

	t.Run("Enabled returns false if no handlers are enabled", func(t *testing.T) {
		h1 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelError})
		h2 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelInfo})

		multi := NewMultiHandler(h1, h2)

		if multi.Enabled(context.Background(), LevelDebug) {
			t.Error("Enabled should be false for DEBUG level, but got true")
		}
	})

	t.Run("WithAttrs propagates attributes to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, nil)
		h2 := NewJSONHandler(&buf2, nil)

		multi := NewMultiHandler(h1, h2).WithAttrs([]Attr{String("request_id", "123")})
		logger := New(multi)

		logger.Info("request processed")

		checkLogOutput(t, buf1.String(), "time="+textTimeRE+` level=INFO msg="request processed" request_id=123`)
		checkLogOutput(t, buf2.String(), `{"time":"`+jsonTimeRE+`","level":"INFO","msg":"request processed","request_id":"123"}`)
	})

	t.Run("WithGroup propagates group to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, &HandlerOptions{AddSource: false})
		h2 := NewJSONHandler(&buf2, &HandlerOptions{AddSource: false})

		multi := NewMultiHandler(h1, h2).WithGroup("req")
		logger := New(multi)

		logger.Info("user login", "user_id", 42)

		checkLogOutput(t, buf1.String(), "time="+textTimeRE+` level=INFO msg="user login" req.user_id=42`)
		checkLogOutput(t, buf2.String(), `{"time":"`+jsonTimeRE+`","level":"INFO","msg":"user login","req":{"user_id":42}}`)
	})

	t.Run("Handle propagates errors from handlers", func(t *testing.T) {
		errFail := errors.New("mock failing")

		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, nil)
		h2 := &mockFailingHandler{Handler: NewJSONHandler(&buf2, nil), err: errFail}

		multi := NewMultiHandler(h2, h1)

		err := multi.Handle(context.Background(), NewRecord(time.Now(), LevelInfo, "test message", 0))
		if !errors.Is(err, errFail) {
			t.Errorf("Expected error: %v, but got: %v", errFail, err)
		}

		checkLogOutput(t, buf1.String(), "time="+textTimeRE+` level=INFO msg="test message"`)
		checkLogOutput(t, buf2.String(), `{"time":"`+jsonTimeRE+`","level":"INFO","msg":"test message"}`)
	})

	t.Run("Handle with no handlers", func(t *testing.T) {
		multi := NewMultiHandler()
		logger := New(multi)

		logger.Info("nothing")

		err := multi.Handle(context.Background(), NewRecord(time.Now(), LevelInfo, "test", 0))
		if err != nil {
			t.Errorf("Handle with no sub-handlers should return nil, but got: %v", err)
		}
	})
}

// Test that NewMultiHandler copies the input slice and is insulated from future modification.
func TestNewMultiHandlerCopy(t *testing.T) {
	var buf1 bytes.Buffer
	h1 := NewTextHandler(&buf1, nil)
	slice := []Handler{h1}
	multi := NewMultiHandler(slice...)
	slice[0] = nil

	err := multi.Handle(context.Background(), NewRecord(time.Now(), LevelInfo, "test message", 0))
	if err != nil {
		t.Errorf("Expected nil error, but got: %v", err)
	}
	checkLogOutput(t, buf1.String(), "time="+textTimeRE+` level=INFO msg="test message"`)
}
