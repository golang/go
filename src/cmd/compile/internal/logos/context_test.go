// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

const contextServiceSrc = `package p

import "context"

type Secret struct {
	Token string
}

type Config struct {
	Addr string
	Key  Secret
}

type Service struct {
	cfg  Config
	name string
}

func NewService(cfg Config) *Service {
	return &Service{cfg: cfg, name: "built-by-newservice"}
}

func (s *Service) Start(ctx context.Context) error {
	if s.name == "" {
		return errServiceMissing
	}
	return nil
}

func (s *Service) Stop() error {
	s.name = "stopped-by-stop"
	return nil
}
`

const contextStoreSrc = `package q

import "context"

type Options struct {
	Addr string
}

type Store struct {
	opts Options
	tag  string
}

func Open(opts Options) *Store {
	return &Store{opts: opts, tag: "opened-by-open"}
}

func (s *Store) Get(ctx context.Context, key string) (string, error) {
	if key == "" {
		return "", errStoreEmpty
	}
	return s.tag, nil
}

func (s *Store) Close() error {
	s.tag = "closed-by-close"
	return nil
}
`

func parseContextFile(t *testing.T, name, src string, mode syntax.Mode) *syntax.File {
	t.Helper()
	base := syntax.NewFileBase(name)
	file, err := syntax.Parse(base, strings.NewReader(src), func(err error) { t.Fatalf("parse: %v", err) }, nil, mode)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return file
}

func contextJoined(t *testing.T, name, src, symbol string, mode syntax.Mode) string {
	t.Helper()
	file := parseContextFile(t, name, src, mode)
	lines, err := ContextLines([]*syntax.File{file}, symbol)
	if err != nil {
		t.Fatalf("ContextLines: %v", err)
	}
	return strings.Join(lines, "\n")
}

func TestContextLgoService(t *testing.T) {
	out := contextJoined(t, "service.lgo", contextServiceSrc, "Service", syntax.LogosMode)
	for _, want := range []string{
		"type Service struct",
		"NewService(cfg Config)",
		"Start(ctx context.Context)",
		"Stop()",
		"type Config struct",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("context lacks %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{
		"built-by-newservice",
		"stopped-by-stop",
		"type Secret struct",
	} {
		if strings.Contains(out, banned) {
			t.Errorf("context must not contain %q:\n%s", banned, out)
		}
	}
}

func TestContextGoStore(t *testing.T) {
	out := contextJoined(t, "store.go", contextStoreSrc, "Store", 0)
	for _, want := range []string{
		"type Store struct",
		"Open(opts Options)",
		"Get(ctx context.Context, key string)",
		"Close()",
		"type Options struct",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("context lacks %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{
		"opened-by-open",
		"closed-by-close",
	} {
		if strings.Contains(out, banned) {
			t.Errorf("context must not contain %q:\n%s", banned, out)
		}
	}
}

func TestContextUnknownSymbol(t *testing.T) {
	file := parseContextFile(t, "service.lgo", contextServiceSrc, syntax.LogosMode)
	_, err := ContextLines([]*syntax.File{file}, "Missing")
	if err == nil {
		t.Fatal("ContextLines succeeded for unknown symbol")
	}
	want := "logos context: p has no symbol Missing"
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err.Error(), want)
	}
}

func TestContextFuncSymbol(t *testing.T) {
	out := contextJoined(t, "service.lgo", contextServiceSrc, "NewService", syntax.LogosMode)
	if !strings.Contains(out, "NewService(cfg Config)") {
		t.Errorf("context lacks func signature:\n%s", out)
	}
	if strings.Contains(out, "built-by-newservice") {
		t.Errorf("context must not contain body text:\n%s", out)
	}
}

func TestTruncateLines(t *testing.T) {
	mk := func(n int) []string {
		lines := make([]string, n)
		for i := range lines {
			lines[i] = "line"
		}
		return lines
	}
	if got := TruncateLines(mk(99), 100); len(got) != 99 {
		t.Errorf("99 lines -> %d", len(got))
	}
	if got := TruncateLines(mk(100), 100); len(got) != 100 {
		t.Errorf("100 lines -> %d", len(got))
	}
	if got := TruncateLines(mk(101), 100); len(got) != 100 {
		t.Errorf("101 lines -> %d", len(got))
	}
}

func TestContextLineCap(t *testing.T) {
	var b strings.Builder
	b.WriteString("package big\n\ntype Big struct {\n\tid int\n}\n")
	for range 150 {
		b.WriteString("func (b *Big) M() {}\n")
	}
	file := parseContextFile(t, "big.lgo", b.String(), syntax.LogosMode)
	lines, err := ContextLines([]*syntax.File{file}, "Big")
	if err != nil {
		t.Fatalf("ContextLines: %v", err)
	}
	if len(lines) != 100 {
		t.Fatalf("lines = %d, want 100", len(lines))
	}
}
