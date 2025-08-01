// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// mockFailingHandler is a handler that always returns an error from its Handle method.
type mockFailingHandler struct {
	Handler
	err error
}

func (h *mockFailingHandler) Handle(ctx context.Context, r Record) error {
	// It still calls the underlying handler's Handle method to ensure the log can be processed.
	_ = h.Handler.Handle(ctx, r)
	// But it always returns a predefined error.
	return h.err
}

func TestMultiHandler(t *testing.T) {
	ctx := context.Background()

	t.Run("Handle sends log to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, nil)
		h2 := NewJSONHandler(&buf2, nil)

		multi := MultiHandler(h1, h2)
		logger := New(multi)

		logger.Info("hello world", "user", "test")

		// Check the output of the Text handler.
		output1 := buf1.String()
		if !strings.Contains(output1, `level=INFO`) ||
			!strings.Contains(output1, `msg="hello world"`) ||
			!strings.Contains(output1, `user=test`) {
			t.Errorf("Text handler did not receive the correct log message. Got: %s", output1)
		}

		// Check the output of the JSON handle.
		output2 := buf2.String()
		if !strings.Contains(output2, `"level":"INFO"`) ||
			!strings.Contains(output2, `"msg":"hello world"`) ||
			!strings.Contains(output2, `"user":"test"`) {
			t.Errorf("JSON handler did not receive the correct log message. Got: %s", output2)
		}
	})

	t.Run("Enabled returns true if any handler is enabled", func(t *testing.T) {
		h1 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelError})
		h2 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelInfo})

		multi := MultiHandler(h1, h2)

		if !multi.Enabled(ctx, LevelInfo) {
			t.Error("Enabled should be true for INFO level, but got false")
		}
		if !multi.Enabled(ctx, LevelError) {
			t.Error("Enabled should be true for ERROR level, but got false")
		}
	})

	t.Run("Enabled returns false if no handlers are enabled", func(t *testing.T) {
		h1 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelError})
		h2 := NewTextHandler(&bytes.Buffer{}, &HandlerOptions{Level: LevelInfo})

		multi := MultiHandler(h1, h2)

		if multi.Enabled(ctx, LevelDebug) {
			t.Error("Enabled should be false for DEBUG level, but got true")
		}
	})

	t.Run("WithAttrs propagates attributes to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, nil)
		h2 := NewJSONHandler(&buf2, nil)

		multi := MultiHandler(h1, h2).WithAttrs([]Attr{String("request_id", "123")})
		logger := New(multi)

		logger.Info("request processed")

		// Check if the Text handler contains the attribute.
		if !strings.Contains(buf1.String(), "request_id=123") {
			t.Errorf("Text handler output missing attribute. Got: %s", buf1.String())
		}

		// Check if the JSON handler contains the attribute.
		if !strings.Contains(buf2.String(), `"request_id":"123"`) {
			t.Errorf("JSON handler output missing attribute. Got: %s", buf2.String())
		}
	})

	t.Run("WithGroup propagates group to all handlers", func(t *testing.T) {
		var buf1, buf2 bytes.Buffer
		h1 := NewTextHandler(&buf1, &HandlerOptions{AddSource: false})
		h2 := NewJSONHandler(&buf2, &HandlerOptions{AddSource: false})

		multi := MultiHandler(h1, h2).WithGroup("req")
		logger := New(multi)

		logger.Info("user login", "user_id", 42)

		// Check if the Text handler contains the group.
		expectedText := "req.user_id=42"
		if !strings.Contains(buf1.String(), expectedText) {
			t.Errorf("Text handler output missing group. Expected to contain %q, Got: %s", expectedText, buf1.String())
		}

		// Check if the JSON handler contains the group.
		expectedJSON := `"req":{"user_id":42}`
		if !strings.Contains(buf2.String(), expectedJSON) {
			t.Errorf("JSON handler output missing group. Expected to contain %q, Got: %s", expectedJSON, buf2.String())
		}
	})

	t.Run("Handle propagates errors from handlers", func(t *testing.T) {
		var buf bytes.Buffer
		h1 := NewTextHandler(&buf, nil)

		// Simulate a handler that will fail.
		errFail := errors.New("fake fail")
		h2 := &mockFailingHandler{
			Handler: NewTextHandler(&bytes.Buffer{}, nil),
			err:     errFail,
		}

		multi := MultiHandler(h1, h2)

		err := multi.Handle(ctx, NewRecord(time.Now(), LevelInfo, "test message", 0))

		// Check if the error was returned correctly.
		if err == nil {
			t.Fatal("Expected an error from Handle, but got nil")
		}
		if !errors.Is(err, errFail) {
			t.Errorf("Expected error: %v, but got: %v", errFail, err)
		}

		// Also, check that the successful handler still output the log.
		if !strings.Contains(buf.String(), "test message") {
			t.Error("The successful handler should still have processed the log")
		}
	})

	t.Run("Handle with no handlers", func(t *testing.T) {
		// Create an empty multi-handler.
		multi := MultiHandler()
		logger := New(multi)

		// This should be safe to call and do nothing.
		logger.Info("this is nothing")

		// Calling Handle directly should also be safe.
		err := multi.Handle(ctx, NewRecord(time.Now(), LevelInfo, "test", 0))
		if err != nil {
			t.Errorf("Handle with no sub-handlers should return nil, but got: %v", err)
		}
	})
}
