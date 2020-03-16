package protocol

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/telemetry/event"
	"golang.org/x/tools/internal/xcontext"
)

type contextKey int

const (
	clientKey = contextKey(iota)
)

func WithClient(ctx context.Context, client Client) context.Context {
	return context.WithValue(ctx, clientKey, client)
}

func LogEvent(ctx context.Context, ev event.Event) (context.Context, event.Event) {
	if !ev.IsLog() {
		return ctx, ev
	}
	client, ok := ctx.Value(clientKey).(Client)
	if !ok {
		return ctx, ev
	}
	msg := &LogMessageParams{Type: Info, Message: fmt.Sprint(ev)}
	if ev.Error != nil {
		msg.Type = Error
	}
	go client.LogMessage(xcontext.Detach(ctx), msg)
	return ctx, ev
}
