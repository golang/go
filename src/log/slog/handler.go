// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"context"
	"fmt"
	"io"
	"log/slog/internal/buffer"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"time"
)

// A Handler handles log records produced by a Logger.
//
// A typical handler may print log records to standard error,
// or write them to a file or database, or perhaps augment them
// with additional attributes and pass them on to another handler.
//
// Any of the Handler's methods may be called concurrently with itself
// or with other methods. It is the responsibility of the Handler to
// manage this concurrency.
//
// Users of the slog package should not invoke Handler methods directly.
// They should use the methods of [Logger] instead.
//
// Before implementing your own handler, consult https://go.dev/s/slog-handler-guide.
type Handler interface {
	// Enabled reports whether the handler handles records at the given level.
	// The handler ignores records whose level is lower.
	// It is called early, before any arguments are processed,
	// to save effort if the log event should be discarded.
	// If called from a Logger method, the first argument is the context
	// passed to that method, or context.Background() if nil was passed
	// or the method does not take a context.
	// The context is passed so Enabled can use its values
	// to make a decision.
	Enabled(context.Context, Level) bool

	// Handle handles the Record.
	// It will only be called when Enabled returns true.
	// The Context argument is as for Enabled.
	// It is present solely to provide Handlers access to the context's values.
	// Canceling the context should not affect record processing.
	// (Among other things, log messages may be necessary to debug a
	// cancellation-related problem.)
	//
	// Handle methods that produce output should observe the following rules:
	//   - If r.Time is the zero time, ignore the time.
	//   - If r.PC is zero, ignore it.
	//   - Attr's values should be resolved.
	//   - If an Attr's key and value are both the zero value, ignore the Attr.
	//     This can be tested with attr.Equal(Attr{}).
	//   - If a group's key is empty, inline the group's Attrs.
	//   - If a group has no Attrs (even if it has a non-empty key),
	//     ignore it.
	//
	// [Logger] discards any errors from Handle. Wrap the Handle method to
	// process any errors from Handlers.
	Handle(context.Context, Record) error

	// WithAttrs returns a new Handler whose attributes consist of
	// both the receiver's attributes and the arguments.
	// The Handler owns the slice: it may retain, modify or discard it.
	WithAttrs(attrs []Attr) Handler

	// WithGroup returns a new Handler with the given group appended to
	// the receiver's existing groups.
	// The keys of all subsequent attributes, whether added by With or in a
	// Record, should be qualified by the sequence of group names.
	//
	// How this qualification happens is up to the Handler, so long as
	// this Handler's attribute keys differ from those of another Handler
	// with a different sequence of group names.
	//
	// A Handler should treat WithGroup as starting a Group of Attrs that ends
	// at the end of the log event. That is,
	//
	//     logger.WithGroup("s").LogAttrs(ctx, level, msg, slog.Int("a", 1), slog.Int("b", 2))
	//
	// should behave like
	//
	//     logger.LogAttrs(ctx, level, msg, slog.Group("s", slog.Int("a", 1), slog.Int("b", 2)))
	//
	// If the name is empty, WithGroup returns the receiver.
	WithGroup(name string) Handler
}

type defaultHandler struct {
	ch *commonHandler
	// internal.DefaultOutput, except for testing
	output func(pc uintptr, data []byte) error
}

func newDefaultHandler(output func(uintptr, []byte) error) *defaultHandler {
	return &defaultHandler{
		ch:     &commonHandler{json: false},
		output: output,
	}
}

func (*defaultHandler) Enabled(_ context.Context, l Level) bool {
	return l >= logLoggerLevel.Level()
}

// Collect the level, attributes and message in a string and
// write it with the default log.Logger.
// Let the log.Logger handle time and file/line.
func (h *defaultHandler) Handle(ctx context.Context, r Record) error {
	buf := buffer.New()
	buf.WriteString(r.Level.String())
	buf.WriteByte(' ')
	buf.WriteString(r.Message)
	state := h.ch.newHandleState(buf, true, " ")
	defer state.free()
	state.appendNonBuiltIns(r)
	return h.output(r.PC, *buf)
}

func (h *defaultHandler) WithAttrs(as []Attr) Handler {
	return &defaultHandler{h.ch.withAttrs(as), h.output}
}

func (h *defaultHandler) WithGroup(name string) Handler {
	return &defaultHandler{h.ch.withGroup(name), h.output}
}

// HandlerOptions are options for a [TextHandler] or [JSONHandler].
// A zero HandlerOptions consists entirely of default values.
type HandlerOptions struct {
	// AddSource causes the handler to compute the source code position
	// of the log statement and add a SourceKey attribute to the output.
	AddSource bool

	// Level reports the minimum record level that will be logged.
	// The handler discards records with lower levels.
	// If Level is nil, the handler assumes LevelInfo.
	// The handler calls Level.Level for each record processed;
	// to adjust the minimum level dynamically, use a LevelVar.
	Level Leveler

	// ReplaceAttr is called to rewrite each non-group attribute before it is logged.
	// The attribute's value has been resolved (see [Value.Resolve]).
	// If ReplaceAttr returns a zero Attr, the attribute is discarded.
	//
	// The built-in attributes with keys "time", "level", "source", and "msg"
	// are passed to this function, except that time is omitted
	// if zero, and source is omitted if AddSource is false.
	//
	// The first argument is a list of currently open groups that contain the
	// Attr. It must not be retained or modified. ReplaceAttr is never called
	// for Group attributes, only their contents. For example, the attribute
	// list
	//
	//     Int("a", 1), Group("g", Int("b", 2)), Int("c", 3)
	//
	// results in consecutive calls to ReplaceAttr with the following arguments:
	//
	//     nil, Int("a", 1)
	//     []string{"g"}, Int("b", 2)
	//     nil, Int("c", 3)
	//
	// ReplaceAttr can be used to change the default keys of the built-in
	// attributes, convert types (for example, to replace a `time.Time` with the
	// integer seconds since the Unix epoch), sanitize personal information, or
	// remove attributes from the output.
	ReplaceAttr func(groups []string, a Attr) Attr
}

// Keys for "built-in" attributes.
const (
	// TimeKey is the key used by the built-in handlers for the time
	// when the log method is called. The associated Value is a [time.Time].
	TimeKey = "time"
	// LevelKey is the key used by the built-in handlers for the level
	// of the log call. The associated value is a [Level].
	LevelKey = "level"
	// MessageKey is the key used by the built-in handlers for the
	// message of the log call. The associated value is a string.
	MessageKey = "msg"
	// SourceKey is the key used by the built-in handlers for the source file
	// and line of the log call. The associated value is a *[Source].
	SourceKey = "source"
)

type commonHandler struct {
	json              bool // true => output JSON; false => output text
	opts              HandlerOptions
	preformattedAttrs []byte
	// groupPrefix is for the text handler only.
	// It holds the prefix for groups that were already pre-formatted.
	// A group will appear here when a call to WithGroup is followed by
	// a call to WithAttrs.
	groupPrefix string
	groups      []string // all groups started from WithGroup
	nOpenGroups int      // the number of groups opened in preformattedAttrs
	mu          *sync.Mutex
	w           io.Writer
}

func (h *commonHandler) clone() *commonHandler {
	// We can't use assignment because we can't copy the mutex.
	return &commonHandler{
		json:              h.json,
		opts:              h.opts,
		preformattedAttrs: slices.Clip(h.preformattedAttrs),
		groupPrefix:       h.groupPrefix,
		groups:            slices.Clip(h.groups),
		nOpenGroups:       h.nOpenGroups,
		w:                 h.w,
		mu:                h.mu, // mutex shared among all clones of this handler
	}
}

// enabled reports whether l is greater than or equal to the
// minimum level.
func (h *commonHandler) enabled(l Level) bool {
	minLevel := LevelInfo
	if h.opts.Level != nil {
		minLevel = h.opts.Level.Level()
	}
	return l >= minLevel
}

func (h *commonHandler) withAttrs(as []Attr) *commonHandler {
	// We are going to ignore empty groups, so if the entire slice consists of
	// them, there is nothing to do.
	if countEmptyGroups(as) == len(as) {
		return h
	}
	h2 := h.clone()
	// Pre-format the attributes as an optimization.
	state := h2.newHandleState((*buffer.Buffer)(&h2.preformattedAttrs), false, "")
	defer state.free()
	state.prefix.WriteString(h.groupPrefix)
	if pfa := h2.preformattedAttrs; len(pfa) > 0 {
		state.sep = h.attrSep()
		if h2.json && pfa[len(pfa)-1] == '{' {
			state.sep = ""
		}
	}
	// Remember the position in the buffer, in case all attrs are empty.
	pos := state.buf.Len()
	state.openGroups()
	if !state.appendAttrs(as) {
		state.buf.SetLen(pos)
	} else {
		// Remember the new prefix for later keys.
		h2.groupPrefix = state.prefix.String()
		// Remember how many opened groups are in preformattedAttrs,
		// so we don't open them again when we handle a Record.
		h2.nOpenGroups = len(h2.groups)
	}
	return h2
}

func (h *commonHandler) withGroup(name string) *commonHandler {
	h2 := h.clone()
	h2.groups = append(h2.groups, name)
	return h2
}

// handle is the internal implementation of Handler.Handle
// used by TextHandler and JSONHandler.
func (h *commonHandler) handle(r Record) error {
	state := h.newHandleState(buffer.New(), true, "")
	defer state.free()
	if h.json {
		state.buf.WriteByte('{')
	}
	// Built-in attributes. They are not in a group.
	stateGroups := state.groups
	state.groups = nil // So ReplaceAttrs sees no groups instead of the pre groups.
	rep := h.opts.ReplaceAttr
	// time
	if !r.Time.IsZero() {
		key := TimeKey
		val := r.Time.Round(0) // strip monotonic to match Attr behavior
		if rep == nil {
			state.appendKey(key)
			state.appendTime(val)
		} else {
			state.appendAttr(Time(key, val))
		}
	}
	// level
	key := LevelKey
	val := r.Level
	if rep == nil {
		state.appendKey(key)
		state.appendString(val.String())
	} else {
		state.appendAttr(Any(key, val))
	}
	// source
	if h.opts.AddSource {
		src := r.Source()
		if src == nil {
			src = &Source{}
		}
		state.appendAttr(Any(SourceKey, src))
	}
	key = MessageKey
	msg := r.Message
	if rep == nil {
		state.appendKey(key)
		state.appendString(msg)
	} else {
		state.appendAttr(String(key, msg))
	}
	state.groups = stateGroups // Restore groups passed to ReplaceAttrs.
	state.appendNonBuiltIns(r)
	state.buf.WriteByte('\n')

	h.mu.Lock()
	defer h.mu.Unlock()
	_, err := h.w.Write(*state.buf)
	return err
}

func (s *handleState) appendNonBuiltIns(r Record) {
	// preformatted Attrs
	if pfa := s.h.preformattedAttrs; len(pfa) > 0 {
		s.buf.WriteString(s.sep)
		s.buf.Write(pfa)
		s.sep = s.h.attrSep()
		if s.h.json && pfa[len(pfa)-1] == '{' {
			s.sep = ""
		}
	}
	// Attrs in Record -- unlike the built-in ones, they are in groups started
	// from WithGroup.
	// If the record has no Attrs, don't output any groups.
	nOpenGroups := s.h.nOpenGroups
	if r.NumAttrs() > 0 {
		s.prefix.WriteString(s.h.groupPrefix)
		// The group may turn out to be empty even though it has attrs (for
		// example, ReplaceAttr may delete all the attrs).
		// So remember where we are in the buffer, to restore the position
		// later if necessary.
		pos := s.buf.Len()
		s.openGroups()
		nOpenGroups = len(s.h.groups)
		empty := true
		r.Attrs(func(a Attr) bool {
			if s.appendAttr(a) {
				empty = false
			}
			return true
		})
		if empty {
			s.buf.SetLen(pos)
			nOpenGroups = s.h.nOpenGroups
		}
	}
	if s.h.json {
		// Close all open groups.
		for range s.h.groups[:nOpenGroups] {
			s.buf.WriteByte('}')
		}
		// Close the top-level object.
		s.buf.WriteByte('}')
	}
}

// attrSep returns the separator between attributes.
func (h *commonHandler) attrSep() string {
	if h.json {
		return ","
	}
	return " "
}

// handleState holds state for a single call to commonHandler.handle.
// The initial value of sep determines whether to emit a separator
// before the next key, after which it stays true.
type handleState struct {
	h       *commonHandler
	buf     *buffer.Buffer
	freeBuf bool           // should buf be freed?
	sep     string         // separator to write before next key
	prefix  *buffer.Buffer // for text: key prefix
	groups  *[]string      // pool-allocated slice of active groups, for ReplaceAttr
}

var groupPool = sync.Pool{New: func() any {
	s := make([]string, 0, 10)
	return &s
}}

func (h *commonHandler) newHandleState(buf *buffer.Buffer, freeBuf bool, sep string) handleState {
	s := handleState{
		h:       h,
		buf:     buf,
		freeBuf: freeBuf,
		sep:     sep,
		prefix:  buffer.New(),
	}
	if h.opts.ReplaceAttr != nil {
		s.groups = groupPool.Get().(*[]string)
		*s.groups = append(*s.groups, h.groups[:h.nOpenGroups]...)
	}
	return s
}

func (s *handleState) free() {
	if s.freeBuf {
		s.buf.Free()
	}
	if gs := s.groups; gs != nil {
		*gs = (*gs)[:0]
		groupPool.Put(gs)
	}
	s.prefix.Free()
}

func (s *handleState) openGroups() {
	for _, n := range s.h.groups[s.h.nOpenGroups:] {
		s.openGroup(n)
	}
}

// Separator for group names and keys.
const keyComponentSep = '.'

// openGroup starts a new group of attributes
// with the given name.
func (s *handleState) openGroup(name string) {
	if s.h.json {
		s.appendKey(name)
		s.buf.WriteByte('{')
		s.sep = ""
	} else {
		s.prefix.WriteString(name)
		s.prefix.WriteByte(keyComponentSep)
	}
	// Collect group names for ReplaceAttr.
	if s.groups != nil {
		*s.groups = append(*s.groups, name)
	}
}

// closeGroup ends the group with the given name.
func (s *handleState) closeGroup(name string) {
	if s.h.json {
		s.buf.WriteByte('}')
	} else {
		(*s.prefix) = (*s.prefix)[:len(*s.prefix)-len(name)-1 /* for keyComponentSep */]
	}
	s.sep = s.h.attrSep()
	if s.groups != nil {
		*s.groups = (*s.groups)[:len(*s.groups)-1]
	}
}

// appendAttrs appends the slice of Attrs.
// It reports whether something was appended.
func (s *handleState) appendAttrs(as []Attr) bool {
	nonEmpty := false
	for _, a := range as {
		if s.appendAttr(a) {
			nonEmpty = true
		}
	}
	return nonEmpty
}

// appendAttr appends the Attr's key and value.
// It handles replacement and checking for an empty key.
// It reports whether something was appended.
func (s *handleState) appendAttr(a Attr) bool {
	a.Value = a.Value.Resolve()
	if rep := s.h.opts.ReplaceAttr; rep != nil && a.Value.Kind() != KindGroup {
		var gs []string
		if s.groups != nil {
			gs = *s.groups
		}
		// a.Value is resolved before calling ReplaceAttr, so the user doesn't have to.
		a = rep(gs, a)
		// The ReplaceAttr function may return an unresolved Attr.
		a.Value = a.Value.Resolve()
	}
	// Elide empty Attrs.
	if a.isEmpty() {
		return false
	}
	// Special case: Source.
	if v := a.Value; v.Kind() == KindAny {
		if src, ok := v.Any().(*Source); ok {
			if src.isEmpty() {
				return false
			}
			if s.h.json {
				a.Value = src.group()
			} else {
				a.Value = StringValue(fmt.Sprintf("%s:%d", src.File, src.Line))
			}
		}
	}
	if a.Value.Kind() == KindGroup {
		attrs := a.Value.Group()
		// Output only non-empty groups.
		if len(attrs) > 0 {
			// The group may turn out to be empty even though it has attrs (for
			// example, ReplaceAttr may delete all the attrs).
			// So remember where we are in the buffer, to restore the position
			// later if necessary.
			pos := s.buf.Len()
			// Inline a group with an empty key.
			if a.Key != "" {
				s.openGroup(a.Key)
			}
			if !s.appendAttrs(attrs) {
				s.buf.SetLen(pos)
				return false
			}
			if a.Key != "" {
				s.closeGroup(a.Key)
			}
		}
	} else {
		s.appendKey(a.Key)
		s.appendValue(a.Value)
	}
	return true
}

func (s *handleState) appendError(err error) {
	s.appendString(fmt.Sprintf("!ERROR:%v", err))
}

func (s *handleState) appendKey(key string) {
	s.buf.WriteString(s.sep)
	if s.prefix != nil && len(*s.prefix) > 0 {
		s.appendTwoStrings(string(*s.prefix), key)
	} else {
		s.appendString(key)
	}
	if s.h.json {
		s.buf.WriteByte(':')
	} else {
		s.buf.WriteByte('=')
	}
	s.sep = s.h.attrSep()
}

// appendTwoStrings implements appendString(prefix + key), but faster.
func (s *handleState) appendTwoStrings(x, y string) {
	buf := *s.buf
	switch {
	case s.h.json:
		buf.WriteByte('"')
		buf = appendEscapedJSONString(buf, x)
		buf = appendEscapedJSONString(buf, y)
		buf.WriteByte('"')
	case !needsQuoting(x) && !needsQuoting(y):
		buf.WriteString(x)
		buf.WriteString(y)
	default:
		buf = strconv.AppendQuote(buf, x+y)
	}
	*s.buf = buf
}

func (s *handleState) appendString(str string) {
	if s.h.json {
		s.buf.WriteByte('"')
		*s.buf = appendEscapedJSONString(*s.buf, str)
		s.buf.WriteByte('"')
	} else {
		// text
		if needsQuoting(str) {
			*s.buf = strconv.AppendQuote(*s.buf, str)
		} else {
			s.buf.WriteString(str)
		}
	}
}

func (s *handleState) appendValue(v Value) {
	defer func() {
		if r := recover(); r != nil {
			// If it panics with a nil pointer, the most likely cases are
			// an encoding.TextMarshaler or error fails to guard against nil,
			// in which case "<nil>" seems to be the feasible choice.
			//
			// Adapted from the code in fmt/print.go.
			if v := reflect.ValueOf(v.any); v.Kind() == reflect.Pointer && v.IsNil() {
				s.appendString("<nil>")
				return
			}

			// Otherwise just print the original panic message.
			s.appendString(fmt.Sprintf("!PANIC: %v", r))
		}
	}()

	var err error
	if s.h.json {
		err = appendJSONValue(s, v)
	} else {
		err = appendTextValue(s, v)
	}
	if err != nil {
		s.appendError(err)
	}
}

func (s *handleState) appendTime(t time.Time) {
	if s.h.json {
		appendJSONTime(s, t)
	} else {
		*s.buf = appendRFC3339Millis(*s.buf, t)
	}
}

func appendRFC3339Millis(b []byte, t time.Time) []byte {
	// Format according to time.RFC3339Nano since it is highly optimized,
	// but truncate it to use millisecond resolution.
	// Unfortunately, that format trims trailing 0s, so add 1/10 millisecond
	// to guarantee that there are exactly 4 digits after the period.
	const prefixLen = len("2006-01-02T15:04:05.000")
	n := len(b)
	t = t.Truncate(time.Millisecond).Add(time.Millisecond / 10)
	b = t.AppendFormat(b, time.RFC3339Nano)
	b = append(b[:n+prefixLen], b[n+prefixLen+1:]...) // drop the 4th digit
	return b
}

// DiscardHandler discards all log output.
// DiscardHandler.Enabled returns false for all Levels.
var DiscardHandler Handler = discardHandler{}

type discardHandler struct{}

func (dh discardHandler) Enabled(context.Context, Level) bool  { return false }
func (dh discardHandler) Handle(context.Context, Record) error { return nil }
func (dh discardHandler) WithAttrs(attrs []Attr) Handler       { return dh }
func (dh discardHandler) WithGroup(name string) Handler        { return dh }
