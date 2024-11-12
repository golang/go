package slog_test

import (
	"log/slog"
	"log/slog/internal/slogtest"
	"os"
)

func ExampleDiscardHandler() {
	// A slog.TextHandler will output logs
	logger1 := slog.New(slog.NewTextHandler(
		os.Stdout,
		&slog.HandlerOptions{ReplaceAttr: slogtest.RemoveTime},
	))
	logger1.Info("message 1")

	// A slog.DiscardHandler will discard all messages
	logger2 := slog.New(slog.DiscardHandler)
	logger2.Info("message 2")

	// Output:
	// level=INFO msg="message 1"
}
