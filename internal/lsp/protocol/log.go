package protocol

import (
	"context"

	"golang.org/x/tools/internal/lsp/xlog"
)

// logSink implements xlog.Sink in terms of the LogMessage call to a client.
type logSink struct {
	client Client
}

func (s logSink) Log(ctx context.Context, level xlog.Level, message string) {
	typ := Log
	switch level {
	case xlog.ErrorLevel:
		typ = Error
	case xlog.InfoLevel:
		typ = Info
	case xlog.DebugLevel:
		typ = Log
	}
	s.client.LogMessage(ctx, &LogMessageParams{Type: typ, Message: message})
}
