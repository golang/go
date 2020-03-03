package telemetry_test

import (
	"context"
	stdlog "log"
	"strings"
	"testing"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	tellog "golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	teltrace "golang.org/x/tools/internal/telemetry/trace"
)

type Hooks struct {
	A func(ctx context.Context, a *int) (context.Context, func())
	B func(ctx context.Context, b *string) (context.Context, func())
}

var (
	Baseline = Hooks{
		A: func(ctx context.Context, a *int) (context.Context, func()) {
			return ctx, func() {}
		},
		B: func(ctx context.Context, b *string) (context.Context, func()) {
			return ctx, func() {}
		},
	}

	StdLog = Hooks{
		A: func(ctx context.Context, a *int) (context.Context, func()) {
			stdlog.Printf("start A where a=%d", *a)
			return ctx, func() {
				stdlog.Printf("end A where a=%d", *a)
			}
		},
		B: func(ctx context.Context, b *string) (context.Context, func()) {
			stdlog.Printf("start B where b=%q", *b)
			return ctx, func() {
				stdlog.Printf("end B where b=%q", *b)
			}
		},
	}

	Log = Hooks{
		A: func(ctx context.Context, a *int) (context.Context, func()) {
			tellog.Print(ctx, "start A", tag.Of("a", *a))
			return ctx, func() {
				tellog.Print(ctx, "end A", tag.Of("a", *a))
			}
		},
		B: func(ctx context.Context, b *string) (context.Context, func()) {
			tellog.Print(ctx, "start B", tag.Of("b", *b))
			return ctx, func() {
				tellog.Print(ctx, "end B", tag.Of("b", *b))
			}
		},
	}

	Trace = Hooks{
		A: func(ctx context.Context, a *int) (context.Context, func()) {
			return teltrace.StartSpan(ctx, "A")
		},
		B: func(ctx context.Context, b *string) (context.Context, func()) {
			return teltrace.StartSpan(ctx, "B")
		},
	}
)

func Benchmark(b *testing.B) {
	b.Run("Baseline", Baseline.runBenchmark)
	b.Run("StdLog", StdLog.runBenchmark)
	export.SetExporter(nil)
	b.Run("LogNoExporter", Log.runBenchmark)
	b.Run("TraceNoExporter", Trace.runBenchmark)

	export.SetExporter(newExporter())
	b.Run("Log", Log.runBenchmark)
	b.Run("Trace", Trace.runBenchmark)
}

func A(ctx context.Context, hooks Hooks, a int) int {
	ctx, done := hooks.A(ctx, &a)
	defer done()
	if a > 0 {
		a = a * 10
	}
	return B(ctx, hooks, a, "Called from A")
}

func B(ctx context.Context, hooks Hooks, a int, b string) int {
	_, done := hooks.B(ctx, &b)
	defer done()
	b = strings.ToUpper(b)
	if len(b) > 1024 {
		b = strings.ToLower(b)
	}
	return a + len(b)
}

func (hooks Hooks) runBenchmark(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	var acc int
	for i := 0; i < b.N; i++ {
		for _, value := range []int{0, 10, 20, 100, 1000} {
			acc += A(ctx, hooks, value)
		}
	}
}

func init() {
	stdlog.SetOutput(new(noopWriter))
}

type noopWriter int

func (nw *noopWriter) Write(b []byte) (int, error) {
	return len(b), nil
}

type loggingExporter struct {
	logger export.Exporter
}

func newExporter() *loggingExporter {
	return &loggingExporter{
		logger: export.LogWriter(new(noopWriter), false),
	}
}

func (e *loggingExporter) ProcessEvent(ctx context.Context, event telemetry.Event) context.Context {
	export.ContextSpan(ctx, event)
	return e.logger.ProcessEvent(ctx, event)
}

func (e *loggingExporter) Metric(ctx context.Context, data telemetry.MetricData) {
	e.logger.Metric(ctx, data)
}
