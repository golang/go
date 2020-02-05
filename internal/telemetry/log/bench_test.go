package log_test

import (
	"context"
	stdlog "log"
	"strings"
	"testing"

	"golang.org/x/tools/internal/telemetry/export"
	tellog "golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
)

func init() {
	stdlog.SetOutput(new(noopWriter))
}

type noopWriter int

func (nw *noopWriter) Write(b []byte) (int, error) {
	return len(b), nil
}

func A(ctx context.Context, a int) int {
	if a > 0 {
		_ = 10 * 12
	}
	return B(ctx, "Called from A")
}

func B(ctx context.Context, b string) int {
	b = strings.ToUpper(b)
	if len(b) > 1024 {
		b = strings.ToLower(b)
	}
	return len(b)
}

func A_log(ctx context.Context, a int) int {
	if a > 0 {
		tellog.Print(ctx, "a > 0", tag.Of("a", a))
		_ = 10 * 12
	}
	tellog.Print(ctx, "calling b")
	return B_log(ctx, "Called from A")
}

func B_log(ctx context.Context, b string) int {
	b = strings.ToUpper(b)
	tellog.Print(ctx, "b uppercased, so lowercased", tag.Of("len_b", len(b)))
	if len(b) > 1024 {
		b = strings.ToLower(b)
		tellog.Print(ctx, "b > 1024, so lowercased", tag.Of("b", b))
	}
	return len(b)
}

func A_log_stdlib(ctx context.Context, a int) int {
	if a > 0 {
		stdlog.Printf("a > 0 where a=%d", a)
		_ = 10 * 12
	}
	stdlog.Print("calling b")
	return B_log_stdlib(ctx, "Called from A")
}

func B_log_stdlib(ctx context.Context, b string) int {
	b = strings.ToUpper(b)
	stdlog.Printf("b uppercased, so lowercased where len_b=%d", len(b))
	if len(b) > 1024 {
		b = strings.ToLower(b)
		stdlog.Printf("b > 1024, so lowercased where b=%s", b)
	}
	return len(b)
}

var values = []int{0, 10, 20, 100, 1000}

func BenchmarkBaseline(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, value := range values {
			if g := A(ctx, value); g <= 0 {
				b.Fatalf("Unexpected got g(%d) <= 0", g)
			}
		}
	}
}

func BenchmarkLoggingNoExporter(b *testing.B) {
	ctx := context.Background()
	export.SetExporter(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, value := range values {
			if g := A_log(ctx, value); g <= 0 {
				b.Fatalf("Unexpected got g(%d) <= 0", g)
			}
		}
	}
}

func BenchmarkLogging(b *testing.B) {
	ctx := context.Background()
	export.SetExporter(export.LogWriter(new(noopWriter), false))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, value := range values {
			if g := A_log(ctx, value); g <= 0 {
				b.Fatalf("Unexpected got g(%d) <= 0", g)
			}
		}
	}
}

func BenchmarkLoggingStdlib(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, value := range values {
			if g := A_log_stdlib(ctx, value); g <= 0 {
				b.Fatalf("Unexpected got g(%d) <= 0", g)
			}
		}
	}
}
