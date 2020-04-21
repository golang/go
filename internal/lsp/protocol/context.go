package protocol

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/xcontext"
)

type contextKey int

const (
	clientKey = contextKey(iota)
)

func WithClient(ctx context.Context, client Client) context.Context {
	return context.WithValue(ctx, clientKey, client)
}

func LogEvent(ctx context.Context, ev core.Event, tags label.Map) context.Context {
	if !event.IsLog(ev) {
		return ctx
	}
	client, ok := ctx.Value(clientKey).(Client)
	if !ok {
		return ctx
	}
	msg := &LogMessageParams{Type: Info, Message: fmt.Sprint(ev)}
	if event.IsError(ev) {
		msg.Type = Error
	}
	go client.LogMessage(xcontext.Detach(ctx), msg)
	return ctx
}
