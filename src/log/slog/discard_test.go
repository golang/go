package slog

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestDiscardHandler(t *testing.T) {
	ctx := context.Background()
	stdout, stderr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = nil, nil // panic on write
	t.Cleanup(func() {
		os.Stdout, os.Stderr = stdout, stderr
	})

	// Just ensure nothing panics during normal usage
	l := New(DiscardHandler)
	l.Info("msg", "a", 1, "b", 2)
	l.Debug("bg", Int("a", 1), "b", 2)
	l.Warn("w", Duration("dur", 3*time.Second))
	l.Error("bad", "a", 1)
	l.Log(ctx, LevelWarn+1, "w", Int("a", 1), String("b", "two"))
	l.LogAttrs(ctx, LevelInfo+1, "a b c", Int("a", 1), String("b", "two"))
	l.Info("info", "a", []Attr{Int("i", 1)})
	l.Info("info", "a", GroupValue(Int("i", 1)))
}
