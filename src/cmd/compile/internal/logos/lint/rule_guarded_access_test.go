// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"

	"internal/testenv"
)

// checkSrc type-checks src and keeps the source bytes, so the directive
// scanners see the comments.
func checkSrc(t testing.TB, kind Kind, name, src string) *Input {
	in := check(t, kind, name, src)
	in.Sources = map[*syntax.File][]byte{in.Files[0]: []byte(src)}
	return in
}

// pedanticWarn is the policy of a module whose logos.toml raises pedantic.
func pedanticWarn() *Policy {
	p := DefaultPolicy()
	p.groups[Pedantic] = Warn
	return p
}

func TestGuardedAccess(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import "sync"

type stats struct {
	mu sync.Mutex

	//logos:guardedby mu
	total int
}

//logos:guardedby gmu
var gtotal int

var gmu sync.Mutex
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"unlocked read", `
func bad(s *stats) int {
	return s.total
}
`, []string{"18 concurrency::unguarded_access forbid"}},
		{"unlocked write", `
func bad(s *stats) {
	s.total = 1
}
`, []string{"18 concurrency::unguarded_access forbid"}},
		{"held read", `
func good(s *stats) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}
`, nil},
		{"branch merge", `
func merge(s *stats, c bool) {
	if c {
		s.mu.Lock()
	}
	_ = s.total
	if c {
		s.mu.Unlock()
	}
}
`, []string{"21 concurrency::unguarded_access forbid"}},
		{"both branches lock", `
func merge(s *stats, c bool) {
	if c {
		s.mu.Lock()
	} else {
		s.mu.Lock()
	}
	_ = s.total
	if c {
		s.mu.Unlock()
	} else {
		s.mu.Unlock()
	}
}
`, nil},
		{"defer holds to exit", `
func hold(s *stats) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total
}
`, nil},
		{"explicit unlock", `
func drop(s *stats) int {
	s.mu.Lock()
	s.mu.Unlock()
	return s.total
}
`, []string{"20 concurrency::unguarded_access forbid"}},
		{"deferred closure after unlock defer", `
func run(s *stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	defer func() {
		_ = s.total
	}()
}
`, nil},
		{"deferred closure before unlock defer", `
func run(s *stats) {
	s.mu.Lock()
	defer func() {
		_ = s.total
	}()
	defer s.mu.Unlock()
}
`, []string{"20 concurrency::unguarded_access forbid"}},
		{"global guard held", `
func good() {
	gmu.Lock()
	defer gmu.Unlock()
	gtotal++
}
`, nil},
		{"global guard dropped", `
func bad() {
	gtotal++
}
`, []string{"18 concurrency::unguarded_access forbid"}},
		{"allocated local exempt", `
func build() *stats {
	t := &stats{}
	t.total = 3
	return t
}
`, nil},
		{"published local guarded", `
var sink = &stats{}

func publish() {
	t := &stats{}
	sink = t
	t.total = 4
}
`, []string{"22 concurrency::unguarded_access forbid"}},
		{"misplaced directive", `
var x = 1

//logos:guardedby mu

func f() {
	_ = x
}
`, []string{"19 concurrency::unguarded_access forbid"}},
		{"acquires contract", `
type vault struct {
	mu sync.Mutex

	//logos:guardedby mu
	keys int
}

//logos:acquires v.mu
func (v *vault) seal() {}

func use(v *vault) {
	v.seal()
	_ = v.keys
}
`, nil},
		{"releases contract", `
type vault struct {
	mu sync.Mutex

	//logos:guardedby mu
	keys int
}

//logos:releases v.mu
func (v *vault) open() {}

func use(v *vault) {
	v.mu.Lock()
	v.open()
	_ = v.keys
}
`, []string{"30 concurrency::unguarded_access forbid"}},
		{"rwmutex write under read lock", `
type rw struct {
	mu sync.RWMutex

	//logos:guardedby mu
	n int
}

func bad(r *rw) {
	r.mu.RLock()
	r.n = 1
	r.mu.RUnlock()
}
`, []string{"26 concurrency::unguarded_access forbid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name+"/lgo", func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", head+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
		t.Run(tt.name+"/go", func(t *testing.T) {
			want := make([]string, 0, len(tt.want))
			for _, w := range tt.want {
				goW, _ := strings.CutSuffix(w, " forbid")
				want = append(want, goW+" deny")
			}
			got := summarize(Run(checkSrc(t, KindGo, "x.go", head+tt.body)))
			if !slices.Equal(got, want) {
				t.Errorf("results = %q, want %q", got, want)
			}
		})
	}
}

func TestLockRequirement(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import "sync"

type gate struct {
	mu sync.Mutex

	//logos:guardedby mu
	open bool
}

//logos:requires g.mu
func (g *gate) toggle() {
	g.open = !g.open
}

//logos:acquires g.mu
func (g *gate) seal() {}

//logos:releases g.mu
func (g *gate) release() {}
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"unmet", `
func bad(g *gate) {
	g.toggle()
}
`, []string{"24 concurrency::lock_requirement_unmet forbid"}},
		{"met", `
func good(g *gate) {
	g.mu.Lock()
	g.toggle()
	g.mu.Unlock()
}
`, nil},
		{"acquired", `
func viaSeal(g *gate) {
	g.seal()
	g.toggle()
}
`, nil},
		{"released", `
func dropped(g *gate) {
	g.mu.Lock()
	g.release()
	g.toggle()
}
`, []string{"26 concurrency::lock_requirement_unmet forbid"}},
		{"wrong receiver", `
func other(g, h *gate) {
	h.mu.Lock()
	g.toggle()
	h.mu.Unlock()
}
`, []string{"25 concurrency::lock_requirement_unmet forbid"}},
		{"deferred release holds", `
func held(g *gate) {
	g.mu.Lock()
	defer g.release()
	g.toggle()
}
`, nil},
		{"misplaced requires", `
var x = 1

//logos:requires g.mu

func f() {
	_ = x
}
`, []string{"25 concurrency::lock_requirement_unmet forbid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", head+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNonatomicAccess(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import "sync/atomic"

var keepAtomic atomic.Int32

type counter struct {

	//logos:atomic
	hits int64

	//logos:atomic
	drops uint64
}

//logos:atomic
var gtotal int64

func passthrough(*int64) {}
`
	tests := []struct {
		name string
		body string
		want []string
	}{
		{"atomic load", `
func good(c *counter) int64 {
	return atomic.LoadInt64(&c.hits)
}
`, nil},
		{"atomic store", `
func good() {
	atomic.StoreInt64(&gtotal, 1)
}
`, nil},
		{"plain read", `
func bad(c *counter) int64 {
	return c.hits
}
`, []string{"22 concurrency::nonatomic_access forbid"}},
		{"plain write", `
func bad(c *counter) {
	c.drops = 1
}
`, []string{"22 concurrency::nonatomic_access forbid"}},
		{"plain increment", `
func bad(c *counter) {
	c.hits++
}
`, []string{"22 concurrency::nonatomic_access forbid"}},
		{"address leaks", `
func bad(c *counter) {
	passthrough(&c.hits)
}
`, []string{"22 concurrency::nonatomic_access forbid"}},
		{"plain global write", `
func bad() {
	gtotal = 5
}
`, []string{"22 concurrency::nonatomic_access forbid"}},
		{"misplaced directive", `
var x = 1

//logos:atomic

func f() {
	_ = x
}
`, []string{"23 concurrency::nonatomic_access forbid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", head+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}
