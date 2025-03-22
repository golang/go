// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
	"io"
	"log"
	loginternal "log/internal"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// textTimeRE is a regexp to match log timestamps for Text handler.
// This is RFC3339Nano with the fixed 3 digit sub-second precision.
const textTimeRE = `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(Z|[+-]\d{2}:\d{2})`

// jsonTimeRE is a regexp to match log timestamps for Text handler.
// This is RFC3339Nano with an arbitrary sub-second precision.
const jsonTimeRE = `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})`

func TestLogTextHandler(t *testing.T) {
	ctx := context.Background()
	var buf bytes.Buffer

	l := New(NewTextHandler(&buf, nil))

	check := func(want string) {
		t.Helper()
		if want != "" {
			want = "time=" + textTimeRE + " " + want
		}
		checkLogOutput(t, buf.String(), want)
		buf.Reset()
	}

	l.Info("msg", "a", 1, "b", 2)
	check(`level=INFO msg=msg a=1 b=2`)

	// By default, debug messages are not printed.
	l.Debug("bg", Int("a", 1), "b", 2)
	check("")

	l.Warn("w", Duration("dur", 3*time.Second))
	check(`level=WARN msg=w dur=3s`)

	l.Error("bad", "a", 1)
	check(`level=ERROR msg=bad a=1`)

	l.Log(ctx, LevelWarn+1, "w", Int("a", 1), String("b", "two"))
	check(`level=WARN\+1 msg=w a=1 b=two`)

	l.LogAttrs(ctx, LevelInfo+1, "a b c", Int("a", 1), String("b", "two"))
	check(`level=INFO\+1 msg="a b c" a=1 b=two`)

	l.Info("info", "a", []Attr{Int("i", 1)})
	check(`level=INFO msg=info a.i=1`)

	l.Info("info", "a", GroupValue(Int("i", 1)))
	check(`level=INFO msg=info a.i=1`)
}

func TestConnections(t *testing.T) {
	var logbuf, slogbuf bytes.Buffer

	// Revert any changes to the default logger. This is important because other
	// tests might change the default logger using SetDefault. Also ensure we
	// restore the default logger at the end of the test.
	currentLogger := Default()
	currentLogWriter := log.Writer()
	currentLogFlags := log.Flags()
	SetDefault(New(newDefaultHandler(loginternal.DefaultOutput)))
	t.Cleanup(func() {
		SetDefault(currentLogger)
		log.SetOutput(currentLogWriter)
		log.SetFlags(currentLogFlags)
	})

	// The default slog.Logger's handler uses the log package's default output.
	log.SetOutput(&logbuf)
	log.SetFlags(log.Lshortfile &^ log.LstdFlags)
	Info("msg", "a", 1)
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: INFO msg a=1`)
	logbuf.Reset()
	Info("msg", "p", nil)
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: INFO msg p=<nil>`)
	logbuf.Reset()
	var r *regexp.Regexp
	Info("msg", "r", r)
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: INFO msg r=<nil>`)
	logbuf.Reset()
	Warn("msg", "b", 2)
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: WARN msg b=2`)
	logbuf.Reset()
	Error("msg", "err", io.EOF, "c", 3)
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: ERROR msg err=EOF c=3`)

	// Levels below Info are not printed.
	logbuf.Reset()
	Debug("msg", "c", 3)
	checkLogOutput(t, logbuf.String(), "")

	t.Run("wrap default handler", func(t *testing.T) {
		// It should be possible to wrap the default handler and get the right output.
		// This works because the default handler uses the pc in the Record
		// to get the source line, rather than a call depth.
		logger := New(wrappingHandler{Default().Handler()})
		logger.Info("msg", "d", 4)
		checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: INFO msg d=4`)
	})

	// Once slog.SetDefault is called, the direction is reversed: the default
	// log.Logger's output goes through the handler.
	SetDefault(New(NewTextHandler(&slogbuf, &HandlerOptions{AddSource: true})))
	log.Print("msg2")
	checkLogOutput(t, slogbuf.String(), "time="+textTimeRE+` level=INFO source=.*logger_test.go:\d{3}"? msg=msg2`)

	// The default log.Logger always outputs at Info level.
	slogbuf.Reset()
	SetDefault(New(NewTextHandler(&slogbuf, &HandlerOptions{Level: LevelWarn})))
	log.Print("should not appear")
	if got := slogbuf.String(); got != "" {
		t.Errorf("got %q, want empty", got)
	}

	// Setting log's output again breaks the connection.
	logbuf.Reset()
	slogbuf.Reset()
	log.SetOutput(&logbuf)
	log.SetFlags(log.Lshortfile &^ log.LstdFlags)
	log.Print("msg3")
	checkLogOutput(t, logbuf.String(), `logger_test.go:\d+: msg3`)
	if got := slogbuf.String(); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

type wrappingHandler struct {
	h Handler
}

func (h wrappingHandler) Enabled(ctx context.Context, level Level) bool {
	return h.h.Enabled(ctx, level)
}
func (h wrappingHandler) WithGroup(name string) Handler              { return h.h.WithGroup(name) }
func (h wrappingHandler) WithAttrs(as []Attr) Handler                { return h.h.WithAttrs(as) }
func (h wrappingHandler) Handle(ctx context.Context, r Record) error { return h.h.Handle(ctx, r) }

func TestAttrs(t *testing.T) {
	check := func(got []Attr, want ...Attr) {
		t.Helper()
		if !attrsEqual(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}

	l1 := New(&captureHandler{}).With("a", 1)
	l2 := New(l1.Handler()).With("b", 2)
	l2.Info("m", "c", 3)
	h := l2.Handler().(*captureHandler)
	check(h.attrs, Int("a", 1), Int("b", 2))
	check(attrsSlice(h.r), Int("c", 3))
}

func TestCallDepth(t *testing.T) {
	ctx := context.Background()
	h := &captureHandler{}
	var startLine int

	check := func(count int) {
		t.Helper()
		const wantFunc = "log/slog.TestCallDepth"
		const wantFile = "logger_test.go"
		wantLine := startLine + count*2
		got := h.r.Source()
		gotFile := filepath.Base(got.File)
		if got.Function != wantFunc || gotFile != wantFile || got.Line != wantLine {
			t.Errorf("got (%s, %s, %d), want (%s, %s, %d)",
				got.Function, gotFile, got.Line, wantFunc, wantFile, wantLine)
		}
	}

	defer SetDefault(Default()) // restore
	logger := New(h)
	SetDefault(logger)

	// Calls to check must be one line apart.
	// Determine line where calls start.
	f, _ := runtime.CallersFrames([]uintptr{callerPC(2)}).Next()
	startLine = f.Line + 4
	// Do not change the number of lines between here and the call to check(0).

	logger.Log(ctx, LevelInfo, "")
	check(0)
	logger.LogAttrs(ctx, LevelInfo, "")
	check(1)
	logger.Debug("")
	check(2)
	logger.Info("")
	check(3)
	logger.Warn("")
	check(4)
	logger.Error("")
	check(5)
	Debug("")
	check(6)
	Info("")
	check(7)
	Warn("")
	check(8)
	Error("")
	check(9)
	Log(ctx, LevelInfo, "")
	check(10)
	LogAttrs(ctx, LevelInfo, "")
	check(11)
}

func TestCallDepthConnection(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	testenv.MustHaveExec(t)
	ep, err := os.Executable()
	if err != nil {
		t.Fatalf("Executable failed: %v", err)
	}

	tests := []struct {
		name string
		log  func()
	}{
		{"log.Fatal", func() { log.Fatal("log.Fatal") }},
		{"log.Fatalf", func() { log.Fatalf("log.Fatalf") }},
		{"log.Fatalln", func() { log.Fatalln("log.Fatalln") }},
		{"log.Output", func() { log.Output(1, "log.Output") }},
		{"log.Panic", func() { log.Panic("log.Panic") }},
		{"log.Panicf", func() { log.Panicf("log.Panicf") }},
		{"log.Panicln", func() { log.Panicf("log.Panicln") }},
		{"log.Default.Fatal", func() { log.Default().Fatal("log.Default.Fatal") }},
		{"log.Default.Fatalf", func() { log.Default().Fatalf("log.Default.Fatalf") }},
		{"log.Default.Fatalln", func() { log.Default().Fatalln("log.Default.Fatalln") }},
		{"log.Default.Output", func() { log.Default().Output(1, "log.Default.Output") }},
		{"log.Default.Panic", func() { log.Default().Panic("log.Default.Panic") }},
		{"log.Default.Panicf", func() { log.Default().Panicf("log.Default.Panicf") }},
		{"log.Default.Panicln", func() { log.Default().Panicf("log.Default.Panicln") }},
	}

	// calculate the line offset until the first test case
	_, _, line, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("runtime.Caller failed")
	}
	line -= len(tests) + 3

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// inside spawned test executable
			const envVar = "SLOGTEST_CALL_DEPTH_CONNECTION"
			if os.Getenv(envVar) == "1" {
				h := NewTextHandler(os.Stderr, &HandlerOptions{
					AddSource: true,
					ReplaceAttr: func(groups []string, a Attr) Attr {
						if (a.Key == MessageKey || a.Key == SourceKey) && len(groups) == 0 {
							return a
						}
						return Attr{}
					},
				})
				SetDefault(New(h))
				log.SetFlags(log.Lshortfile)
				tt.log()
				os.Exit(1)
			}

			// spawn test executable
			cmd := testenv.Command(t, ep,
				"-test.run=^"+regexp.QuoteMeta(t.Name())+"$",
				"-test.count=1",
			)
			cmd.Env = append(cmd.Environ(), envVar+"=1")

			out, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) {
				t.Fatalf("expected exec.ExitError: %v", err)
			}

			_, firstLine, err := bufio.ScanLines(out, true)
			if err != nil {
				t.Fatalf("failed to split line: %v", err)
			}
			got := string(firstLine)

			want := fmt.Sprintf(
				`source=:0 msg="logger_test.go:%d: %s"`,
				line+i, tt.name,
			)
			if got != want {
				t.Errorf(
					"output from %s() mismatch:\n\t got: %s\n\twant: %s",
					tt.name, got, want,
				)
			}
		})
	}
}

func TestAlloc(t *testing.T) {
	ctx := context.Background()
	dl := New(discardTestHandler{})
	defer SetDefault(Default()) // restore
	SetDefault(dl)

	t.Run("Info", func(t *testing.T) {
		wantAllocs(t, 0, func() { Info("hello") })
	})
	t.Run("Error", func(t *testing.T) {
		wantAllocs(t, 0, func() { Error("hello") })
	})
	t.Run("logger.Info", func(t *testing.T) {
		wantAllocs(t, 0, func() { dl.Info("hello") })
	})
	t.Run("logger.Log", func(t *testing.T) {
		wantAllocs(t, 0, func() { dl.Log(ctx, LevelDebug, "hello") })
	})
	t.Run("2 pairs", func(t *testing.T) {
		s := "abc"
		i := 2000
		wantAllocs(t, 2, func() {
			dl.Info("hello",
				"n", i,
				"s", s,
			)
		})
	})
	t.Run("2 pairs disabled inline", func(t *testing.T) {
		l := New(DiscardHandler)
		s := "abc"
		i := 2000
		wantAllocs(t, 2, func() {
			l.Log(ctx, LevelInfo, "hello",
				"n", i,
				"s", s,
			)
		})
	})
	t.Run("2 pairs disabled", func(t *testing.T) {
		l := New(DiscardHandler)
		s := "abc"
		i := 2000
		wantAllocs(t, 0, func() {
			if l.Enabled(ctx, LevelInfo) {
				l.Log(ctx, LevelInfo, "hello",
					"n", i,
					"s", s,
				)
			}
		})
	})
	t.Run("9 kvs", func(t *testing.T) {
		s := "abc"
		i := 2000
		d := time.Second
		wantAllocs(t, 10, func() {
			dl.Info("hello",
				"n", i, "s", s, "d", d,
				"n", i, "s", s, "d", d,
				"n", i, "s", s, "d", d)
		})
	})
	t.Run("pairs", func(t *testing.T) {
		wantAllocs(t, 0, func() { dl.Info("", "error", io.EOF) })
	})
	t.Run("attrs1", func(t *testing.T) {
		wantAllocs(t, 0, func() { dl.LogAttrs(ctx, LevelInfo, "", Int("a", 1)) })
		wantAllocs(t, 0, func() { dl.LogAttrs(ctx, LevelInfo, "", Any("error", io.EOF)) })
	})
	t.Run("attrs3", func(t *testing.T) {
		wantAllocs(t, 0, func() {
			dl.LogAttrs(ctx, LevelInfo, "hello", Int("a", 1), String("b", "two"), Duration("c", time.Second))
		})
	})
	t.Run("attrs3 disabled", func(t *testing.T) {
		logger := New(DiscardHandler)
		wantAllocs(t, 0, func() {
			logger.LogAttrs(ctx, LevelInfo, "hello", Int("a", 1), String("b", "two"), Duration("c", time.Second))
		})
	})
	t.Run("attrs6", func(t *testing.T) {
		wantAllocs(t, 1, func() {
			dl.LogAttrs(ctx, LevelInfo, "hello",
				Int("a", 1), String("b", "two"), Duration("c", time.Second),
				Int("d", 1), String("e", "two"), Duration("f", time.Second))
		})
	})
	t.Run("attrs9", func(t *testing.T) {
		wantAllocs(t, 1, func() {
			dl.LogAttrs(ctx, LevelInfo, "hello",
				Int("a", 1), String("b", "two"), Duration("c", time.Second),
				Int("d", 1), String("e", "two"), Duration("f", time.Second),
				Int("d", 1), String("e", "two"), Duration("f", time.Second))
		})
	})
}

func TestSetAttrs(t *testing.T) {
	for _, test := range []struct {
		args []any
		want []Attr
	}{
		{nil, nil},
		{[]any{"a", 1}, []Attr{Int("a", 1)}},
		{[]any{"a", 1, "b", "two"}, []Attr{Int("a", 1), String("b", "two")}},
		{[]any{"a"}, []Attr{String(badKey, "a")}},
		{[]any{"a", 1, "b"}, []Attr{Int("a", 1), String(badKey, "b")}},
		{[]any{"a", 1, 2, 3}, []Attr{Int("a", 1), Int(badKey, 2), Int(badKey, 3)}},
	} {
		r := NewRecord(time.Time{}, 0, "", 0)
		r.Add(test.args...)
		got := attrsSlice(r)
		if !attrsEqual(got, test.want) {
			t.Errorf("%v:\ngot  %v\nwant %v", test.args, got, test.want)
		}
	}
}

func TestSetDefault(t *testing.T) {
	// Verify that setting the default to itself does not result in deadlock.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	defer func(w io.Writer) { log.SetOutput(w) }(log.Writer())
	log.SetOutput(io.Discard)
	go func() {
		Info("A")
		SetDefault(Default())
		Info("B")
		cancel()
	}()
	<-ctx.Done()
	if err := ctx.Err(); err != context.Canceled {
		t.Errorf("wanted canceled, got %v", err)
	}
}

// Test defaultHandler minimum level without calling slog.SetDefault.
func TestLogLoggerLevelForDefaultHandler(t *testing.T) {
	// Revert any changes to the default logger, flags, and level of log and slog.
	currentLogLoggerLevel := logLoggerLevel.Level()
	currentLogWriter := log.Writer()
	currentLogFlags := log.Flags()
	t.Cleanup(func() {
		logLoggerLevel.Set(currentLogLoggerLevel)
		log.SetOutput(currentLogWriter)
		log.SetFlags(currentLogFlags)
	})

	var logBuf bytes.Buffer
	log.SetOutput(&logBuf)
	log.SetFlags(0)

	for _, test := range []struct {
		logLevel Level
		logFn    func(string, ...any)
		want     string
	}{
		{LevelDebug, Debug, "DEBUG a"},
		{LevelDebug, Info, "INFO a"},
		{LevelInfo, Debug, ""},
		{LevelInfo, Info, "INFO a"},
	} {
		SetLogLoggerLevel(test.logLevel)
		test.logFn("a")
		checkLogOutput(t, logBuf.String(), test.want)
		logBuf.Reset()
	}
}

// Test handlerWriter minimum level by calling slog.SetDefault.
func TestLogLoggerLevelForHandlerWriter(t *testing.T) {
	removeTime := func(_ []string, a Attr) Attr {
		if a.Key == TimeKey {
			return Attr{}
		}
		return a
	}

	// Revert any changes to the default logger. This is important because other
	// tests might change the default logger using SetDefault. Also ensure we
	// restore the default logger at the end of the test.
	currentLogger := Default()
	currentLogLoggerLevel := logLoggerLevel.Level()
	currentLogWriter := log.Writer()
	currentFlags := log.Flags()
	t.Cleanup(func() {
		SetDefault(currentLogger)
		logLoggerLevel.Set(currentLogLoggerLevel)
		log.SetOutput(currentLogWriter)
		log.SetFlags(currentFlags)
	})

	var logBuf bytes.Buffer
	log.SetOutput(&logBuf)
	log.SetFlags(0)
	SetLogLoggerLevel(LevelError)
	SetDefault(New(NewTextHandler(&logBuf, &HandlerOptions{ReplaceAttr: removeTime})))
	log.Print("error")
	checkLogOutput(t, logBuf.String(), `level=ERROR msg=error`)
}

func TestLoggerError(t *testing.T) {
	var buf bytes.Buffer

	removeTime := func(_ []string, a Attr) Attr {
		if a.Key == TimeKey {
			return Attr{}
		}
		return a
	}
	l := New(NewTextHandler(&buf, &HandlerOptions{ReplaceAttr: removeTime}))
	l.Error("msg", "err", io.EOF, "a", 1)
	checkLogOutput(t, buf.String(), `level=ERROR msg=msg err=EOF a=1`)
	buf.Reset()
	// use local var 'args' to defeat vet check
	args := []any{"err", io.EOF, "a"}
	l.Error("msg", args...)
	checkLogOutput(t, buf.String(), `level=ERROR msg=msg err=EOF !BADKEY=a`)
}

func TestNewLogLogger(t *testing.T) {
	var buf bytes.Buffer
	h := NewTextHandler(&buf, nil)
	ll := NewLogLogger(h, LevelWarn)
	ll.Print("hello")
	checkLogOutput(t, buf.String(), "time="+textTimeRE+` level=WARN msg=hello`)
}

func TestLoggerNoOps(t *testing.T) {
	l := Default()
	if l.With() != l {
		t.Error("wanted receiver, didn't get it")
	}
	if With() != l {
		t.Error("wanted receiver, didn't get it")
	}
	if l.WithGroup("") != l {
		t.Error("wanted receiver, didn't get it")
	}
}

func TestContext(t *testing.T) {
	// Verify that the context argument to log output methods is passed to the handler.
	// Also check the level.
	h := &captureHandler{}
	l := New(h)
	defer SetDefault(Default()) // restore
	SetDefault(l)

	for _, test := range []struct {
		f         func(context.Context, string, ...any)
		wantLevel Level
	}{
		{l.DebugContext, LevelDebug},
		{l.InfoContext, LevelInfo},
		{l.WarnContext, LevelWarn},
		{l.ErrorContext, LevelError},
		{DebugContext, LevelDebug},
		{InfoContext, LevelInfo},
		{WarnContext, LevelWarn},
		{ErrorContext, LevelError},
	} {
		h.clear()
		ctx := context.WithValue(context.Background(), "L", test.wantLevel)

		test.f(ctx, "msg")
		if gv := h.ctx.Value("L"); gv != test.wantLevel || h.r.Level != test.wantLevel {
			t.Errorf("got context value %v, level %s; want %s for both", gv, h.r.Level, test.wantLevel)
		}
	}
}

func checkLogOutput(t *testing.T, got, wantRegexp string) {
	t.Helper()
	got = clean(got)
	wantRegexp = "^" + wantRegexp + "$"
	matched, err := regexp.MatchString(wantRegexp, got)
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("\ngot  %s\nwant %s", got, wantRegexp)
	}
}

// clean prepares log output for comparison.
func clean(s string) string {
	if len(s) > 0 && s[len(s)-1] == '\n' {
		s = s[:len(s)-1]
	}
	return strings.ReplaceAll(s, "\n", "~")
}

type captureHandler struct {
	mu     sync.Mutex
	ctx    context.Context
	r      Record
	attrs  []Attr
	groups []string
}

func (h *captureHandler) Handle(ctx context.Context, r Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.ctx = ctx
	h.r = r
	return nil
}

func (*captureHandler) Enabled(context.Context, Level) bool { return true }

func (c *captureHandler) WithAttrs(as []Attr) Handler {
	c.mu.Lock()
	defer c.mu.Unlock()
	var c2 captureHandler
	c2.r = c.r
	c2.groups = c.groups
	c2.attrs = concat(c.attrs, as)
	return &c2
}

func (c *captureHandler) WithGroup(name string) Handler {
	c.mu.Lock()
	defer c.mu.Unlock()
	var c2 captureHandler
	c2.r = c.r
	c2.attrs = c.attrs
	c2.groups = append(slices.Clip(c.groups), name)
	return &c2
}

func (c *captureHandler) clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ctx = nil
	c.r = Record{}
}

type discardTestHandler struct {
	attrs []Attr
}

func (d discardTestHandler) Enabled(context.Context, Level) bool { return true }
func (discardTestHandler) Handle(context.Context, Record) error  { return nil }
func (d discardTestHandler) WithAttrs(as []Attr) Handler {
	d.attrs = concat(d.attrs, as)
	return d
}
func (h discardTestHandler) WithGroup(name string) Handler {
	return h
}

// concat returns a new slice with the elements of s1 followed
// by those of s2. The slice has no additional capacity.
func concat[T any](s1, s2 []T) []T {
	s := make([]T, len(s1)+len(s2))
	copy(s, s1)
	copy(s[len(s1):], s2)
	return s
}

// This is a simple benchmark. See the benchmarks subdirectory for more extensive ones.
func BenchmarkNopLog(b *testing.B) {
	ctx := context.Background()
	l := New(&captureHandler{})
	b.Run("no attrs", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			l.LogAttrs(ctx, LevelInfo, "msg")
		}
	})
	b.Run("attrs", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			l.LogAttrs(ctx, LevelInfo, "msg", Int("a", 1), String("b", "two"), Bool("c", true))
		}
	})
	b.Run("attrs-parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				l.LogAttrs(ctx, LevelInfo, "msg", Int("a", 1), String("b", "two"), Bool("c", true))
			}
		})
	})
	b.Run("keys-values", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			l.Log(ctx, LevelInfo, "msg", "a", 1, "b", "two", "c", true)
		}
	})
	b.Run("WithContext", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			l.LogAttrs(ctx, LevelInfo, "msg2", Int("a", 1), String("b", "two"), Bool("c", true))
		}
	})
	b.Run("WithContext-parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				l.LogAttrs(ctx, LevelInfo, "msg", Int("a", 1), String("b", "two"), Bool("c", true))
			}
		})
	})
}

// callerPC returns the program counter at the given stack depth.
func callerPC(depth int) uintptr {
	var pcs [1]uintptr
	runtime.Callers(depth, pcs[:])
	return pcs[0]
}

func wantAllocs(t *testing.T, want int, f func()) {
	if race.Enabled || asan.Enabled || msan.Enabled {
		t.Skip("skipping test in race, asan, and msan modes")
	}
	testenv.SkipIfOptimizationOff(t)
	t.Helper()
	got := int(testing.AllocsPerRun(5, f))
	if got != want {
		t.Errorf("got %d allocs, want %d", got, want)
	}
}

// panicTextAndJsonMarshaler is a type that panics in MarshalText and MarshalJSON.
type panicTextAndJsonMarshaler struct {
	msg any
}

func (p panicTextAndJsonMarshaler) MarshalText() ([]byte, error) {
	panic(p.msg)
}

func (p panicTextAndJsonMarshaler) MarshalJSON() ([]byte, error) {
	panic(p.msg)
}

func TestPanics(t *testing.T) {
	// Revert any changes to the default logger. This is important because other
	// tests might change the default logger using SetDefault. Also ensure we
	// restore the default logger at the end of the test.
	currentLogger := Default()
	currentLogWriter := log.Writer()
	currentLogFlags := log.Flags()
	t.Cleanup(func() {
		SetDefault(currentLogger)
		log.SetOutput(currentLogWriter)
		log.SetFlags(currentLogFlags)
	})

	var logBuf bytes.Buffer
	log.SetOutput(&logBuf)
	log.SetFlags(log.Lshortfile &^ log.LstdFlags)

	SetDefault(New(newDefaultHandler(loginternal.DefaultOutput)))
	for _, pt := range []struct {
		in  any
		out string
	}{
		{(*panicTextAndJsonMarshaler)(nil), `logger_test.go:\d+: INFO msg p=<nil>`},
		{panicTextAndJsonMarshaler{io.ErrUnexpectedEOF}, `logger_test.go:\d+: INFO msg p="!PANIC: unexpected EOF"`},
		{panicTextAndJsonMarshaler{"panicking"}, `logger_test.go:\d+: INFO msg p="!PANIC: panicking"`},
		{panicTextAndJsonMarshaler{42}, `logger_test.go:\d+: INFO msg p="!PANIC: 42"`},
	} {
		Info("msg", "p", pt.in)
		checkLogOutput(t, logBuf.String(), pt.out)
		logBuf.Reset()
	}

	SetDefault(New(NewJSONHandler(&logBuf, nil)))
	for _, pt := range []struct {
		in  any
		out string
	}{
		{(*panicTextAndJsonMarshaler)(nil), `{"time":"` + jsonTimeRE + `","level":"INFO","msg":"msg","p":null}`},
		{panicTextAndJsonMarshaler{io.ErrUnexpectedEOF}, `{"time":"` + jsonTimeRE + `","level":"INFO","msg":"msg","p":"!PANIC: unexpected EOF"}`},
		{panicTextAndJsonMarshaler{"panicking"}, `{"time":"` + jsonTimeRE + `","level":"INFO","msg":"msg","p":"!PANIC: panicking"}`},
		{panicTextAndJsonMarshaler{42}, `{"time":"` + jsonTimeRE + `","level":"INFO","msg":"msg","p":"!PANIC: 42"}`},
	} {
		Info("msg", "p", pt.in)
		checkLogOutput(t, logBuf.String(), pt.out)
		logBuf.Reset()
	}
}
