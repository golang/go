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

func LogEvent(ctx context.Context, ev event.Event, tags event.TagMap) context.Context {
	if !ev.IsLog() {
		return ctx
	}
	client, ok := ctx.Value(clientKey).(Client)
	if !ok {
		return ctx
	}
	msg := &LogMessageParams{Type: Info, Message: fmt.Sprint(ev)}
	if event.Err.Get(tags) != nil {
		msg.Type = Error
	}
	go client.LogMessage(xcontext.Detach(ctx), msg)
	return ctx
}
