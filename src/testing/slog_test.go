package testing_test

import (
	"log/slog"
	"os"
	"testing"
)

func TestSlog(t *testing.T) {
	// Slog log lines as they are currently printed out in tests.
	logger1 := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{AddSource: true}))
	logger1.Info("slog logging in parent test")

	/*
		t.Slog() allows:
		- the indentation of slog output to match t.Log() output
		- printing of the output under the correct test header
	*/
	logger2 := t.Slog()
	logger2.Info("t.Slog log in parent test")
	t.Error("t.Log in parent test")

	// Additionally, t.Slog() indents slog output depending on the nesting level of the test.
	t.Run("Subtest", func(t *testing.T) {
		logger3 := t.Slog()
		logger3.Info("t.Slog log in subtest")
		t.Error("t.Log in subtest")

		// Without t.Slog(), the slog log does not take into account the nesting level.
		// This is in addition to the log being printed in the wrong section.
		logger4 := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{AddSource: true}))
		logger4.Info("slog logging in subtest")
	})
}
