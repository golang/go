// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"internal/testenv"
	"slices"
	"strconv"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

// measureFuncs returns the metrics of every function of one source file,
// keyed by name: a declaration by its name, a function literal or lambda by
// "literal" and its order of appearance, from 1.
func measureFuncs(t *testing.T, kind Kind, name, src string) map[string]cogMetrics {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	in := check(t, kind, name, src)
	out := make(map[string]cogMetrics)
	literals := 0
	for _, f := range in.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			fn, ok := cogFunctionOf(n)
			if !ok {
				return true
			}
			key, isDecl := strings.CutPrefix(fn.name, "function ")
			if !isDecl || key == "literal" {
				literals++
				key = "literal" + strconv.Itoa(literals)
			}
			out[key] = fn.measure()
			return true
		})
	}
	return out
}

// wantMetrics is the expected depth, params, lines, and score of a function.
type wantMetrics struct{ depth, params, lines, score int }

// compareFuncMetrics reports each function whose measures differ from want.
func compareFuncMetrics(t *testing.T, label string, got map[string]cogMetrics, wants map[string]wantMetrics) {
	t.Helper()
	for name, w := range wants {
		g, ok := got[name]
		if !ok {
			t.Errorf("%s: no function %s", label, name)
			continue
		}
		if g != cogMetrics(w) {
			t.Errorf("%s: %s = %+v, want %+v", label, name, g, cogMetrics(w))
		}
	}
	for name := range got {
		if _, ok := wants[name]; !ok {
			t.Errorf("%s: unexpected function %s", label, name)
		}
	}
}

const metricsSrc = `package p

type T struct{}

func work()           {}
func test(bool) bool  { return true }
func use(...any)      {}

func empty() {}

func oneIf(a bool) {
	if a {
		work()
	}
}

func ifElse(a bool) {
	if a {
		work()
	} else {
		work()
	}
}

func elseIfs(n int) {
	if n == 1 {
	} else if n == 2 {
	} else if n == 3 {
	} else {
	}
}

func elseBlockIf(a, b bool) {
	if a {
	} else {
		if b {
		}
	}
}

func nestedFor(n int) {
	for i := 0; i < n; i++ {
		if i > 1 {
			for j := 0; j < i; j++ {
				work()
			}
		}
	}
}

func rangeLoop(xs []int) {
	for range xs {
	}
}

func infiniteLoop() {
	for {
		break
	}
}

func switchCases(n int) {
	switch n {
	case 1:
	case 2, 3:
	default:
	}
}

func switchNoCases(n int) {
	switch n {
	default:
	}
}

func switchInFor(n int) {
	for {
		switch n {
		case 1:
		case 2:
		default:
		}
	}
}

func typeSwitch(v any) {
	switch v.(type) {
	case int:
	case string:
	default:
	}
}

func selectCases(a, b chan int) {
	select {
	case <-a:
	case b <- 1:
	default:
	}
}

func gotoLoop(n int) {
	i := 0
loop:
	if i < n {
		i++
		goto loop
	}
}

func labeledContinue(n int) {
outer:
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j > i {
				continue outer
			}
		}
	}
}

func labeledBreak(n int) {
outer:
	for i := 0; i < n; i++ {
		break outer
	}
}

func plainJumps(a bool) {
	for {
		if a {
			break
		}
		continue
	}
}

func boolTrailingOr(a, b, c bool) bool {
	return a && b && c || a
}

func boolSameAnd(a, b, c bool) bool {
	return a && b && c
}

func boolMixed(a, b, c bool) bool {
	return a && b || c
}

func boolTwoChanges(a, b, c, d bool) bool {
	return a && b || c && d
}

func boolOrAnd(a, b, c bool) bool {
	return a || b && c
}

func boolParens(a, b, c, d bool) bool {
	return (a || b) && c && (a || d)
}

func boolInIf(a, b, c bool) {
	if a && b || c {
	}
}

func boolInCall(a, b, c bool) {
	use(a && b || c)
}

func boolSeparateChains(a, b, c bool) {
	use(a && b, b || c)
}

func boolNot(a, b bool) bool {
	return !a && !b
}

func closures(n int) {
	f := func() {
		if n > 0 {
			work()
		}
	}
	f()
}

func closureInIf(n int) {
	if n > 0 {
		f := func() {
			for range n {
			}
		}
		f()
	}
}

func params4(a, b, c, d int) {}

func (T) method(a int) {}

func variadic(a int, rest ...int) {}

func unnamed(int, string) {}

func multiLine(
	a int,
	b int,
) {
	work()
}

func blankAndComments() {
	// one

	work()
}
`

// TestCognitiveMetrics pins the four measures of each construct.
func TestCognitiveMetrics(t *testing.T) {
	wants := map[string]wantMetrics{
		"work":               {0, 0, 1, 0},
		"test":               {0, 1, 1, 0},
		"use":                {0, 1, 1, 0},
		"empty":              {0, 0, 1, 0},
		"oneIf":              {1, 1, 5, 1},
		"ifElse":             {1, 1, 7, 1},
		"elseIfs":            {1, 1, 7, 3},
		"elseBlockIf":        {2, 2, 7, 3},
		"nestedFor":          {3, 1, 9, 6},
		"rangeLoop":          {1, 1, 4, 1},
		"infiniteLoop":       {1, 0, 5, 1},
		"switchCases":        {1, 1, 7, 3},
		"switchNoCases":      {1, 1, 5, 1},
		"switchInFor":        {2, 1, 9, 7},
		"typeSwitch":         {1, 1, 7, 3},
		"selectCases":        {1, 2, 7, 3},
		"gotoLoop":           {1, 1, 8, 3},
		"labeledContinue":    {3, 1, 10, 10},
		"labeledBreak":       {1, 1, 6, 3},
		"plainJumps":         {2, 1, 8, 3},
		"boolTrailingOr":     {0, 3, 3, 1},
		"boolSameAnd":        {0, 3, 3, 0},
		"boolMixed":          {0, 3, 3, 1},
		"boolTwoChanges":     {0, 4, 3, 2},
		"boolOrAnd":          {0, 3, 3, 1},
		"boolParens":         {0, 4, 3, 0},
		"boolInIf":           {1, 3, 4, 2},
		"boolInCall":         {0, 3, 3, 1},
		"boolSeparateChains": {0, 3, 3, 0},
		"boolNot":            {0, 2, 3, 0},
		"closures":           {0, 1, 8, 0},
		"closureInIf":        {1, 1, 9, 1},
		"literal1":           {1, 0, 5, 1},
		"literal2":           {1, 0, 4, 1},
		"params4":            {0, 4, 1, 0},
		"method":             {0, 1, 1, 0},
		"variadic":           {0, 2, 1, 0},
		"unnamed":            {0, 2, 1, 0},
		"multiLine":          {0, 2, 6, 0},
		"blankAndComments":   {0, 0, 5, 0},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		compareFuncMetrics(t, name, measureFuncs(t, kind, name, metricsSrc), wants)
	}
}

const lgoMetricsSrc = `package p

func work() {}

func lambdas() {
	var two func(int, int) int = (a, b) => a + b
	var block func(int) int = x => {
		if x > 0 {
			return x
		}
		return 0
	}
	_, _ = two, block
}

func switchExpr(n int) int {
	return switch n {
	case 1 => 1
	case 2 => 2
	default => 0
	}
}

func switchExprInIf(n int) int {
	if n > 0 {
		return switch n {
		case 1 => 1
		default => 0
		}
	}
	return 0
}

func constIf() {
	const if 1 == 1 {
		work()
	}
}

func constIfChain() {
	const if 1 == 2 {
		work()
	} else const if 1 == 1 {
		work()
	} else {
		work()
	}
}

func constIfNested(n int) {
	if n > 0 {
		const if 1 == 1 {
			work()
		}
	}
}
`

// TestCognitiveMetricsLgo pins the loGo/s forms: a lambda is a function of
// its own, a switch expression is a switch with conditioned arms, and a
// const if is an if.
func TestCognitiveMetricsLgo(t *testing.T) {
	compareFuncMetrics(t, "x.lgo", measureFuncs(t, KindLgo, "x.lgo", lgoMetricsSrc), map[string]wantMetrics{
		"work":           {0, 0, 1, 0},
		"lambdas":        {0, 0, 10, 0},
		"literal1":       {0, 2, 1, 0},
		"literal2":       {1, 1, 6, 1},
		"switchExpr":     {1, 1, 7, 3},
		"switchExprInIf": {2, 1, 9, 5},
		"constIf":        {1, 0, 5, 1},
		"constIfChain":   {1, 0, 9, 2},
		"constIfNested":  {2, 1, 7, 3},
	})
}

// thresholdSrc returns a source with three functions each sized one below,
// at, and one above limit for the measure.
func thresholdSrc(measure string, limit int) string {
	var b strings.Builder
	b.WriteString("package p\n\nfunc work() {}\n\n")
	for i, n := range []int{limit - 1, limit, limit + 1} {
		name := fmt.Sprintf("f%d", i)
		switch measure {
		case "params":
			params := make([]string, n)
			for j := range params {
				params[j] = fmt.Sprintf("p%d int", j)
			}
			fmt.Fprintf(&b, "func %s(%s) {}\n\n", name, strings.Join(params, ", "))
		case "lines":
			fmt.Fprintf(&b, "func %s() {\n%s}\n\n", name, strings.Repeat("\twork()\n", n-2))
		case "depth":
			fmt.Fprintf(&b, "func %s(a bool) {\n", name)
			for j := range n {
				b.WriteString(strings.Repeat("\t", j+1) + "if a {\n")
			}
			for j := n - 1; j >= 0; j-- {
				b.WriteString(strings.Repeat("\t", j+1) + "}\n")
			}
			b.WriteString("}\n\n")
		case "score":
			fmt.Fprintf(&b, "func %s(a bool) {\n%s}\n\n", name, strings.Repeat("\tif a {\n\t}\n", n))
		}
	}
	return b.String()
}

// TestCognitiveOverflowThresholds runs each limit at a selected N: a
// function at N-1 and at N is clean and one at N+1 is flagged, and the
// message names the selected limit.
func TestCognitiveOverflowThresholds(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, tc := range []struct {
		measure, key, label string
		limit               int
	}{
		{"params", "max_function_params", "parameter count", 6},
		{"params", "max_function_params", "parameter count", 1},
		{"lines", "max_function_lines", "line count", 10},
		{"lines", "max_function_lines", "line count", 80},
		{"depth", "max_nesting_depth", "nesting depth", 2},
		{"depth", "max_nesting_depth", "nesting depth", 6},
		{"score", "max_cognitive_complexity", "cognitive score", 5},
		{"score", "max_cognitive_complexity", "cognitive score", 20},
	} {
		src := thresholdSrc(tc.measure, tc.limit)
		thresholds := fmt.Sprintf(`%q:%d`, tc.key, tc.limit)
		if tc.measure == "depth" {
			thresholds += `,"max_cognitive_complexity":1000`
		}
		p, err := ParsePolicy([]byte(`{"version":1,"thresholds":{` + thresholds + `}}`))
		if err != nil {
			t.Fatal(err)
		}
		for _, kind := range []Kind{KindGo, KindLgo} {
			name := "x.go"
			if kind == KindLgo {
				name = "x.lgo"
			}
			in := check(t, kind, name, src)
			in.Policy = p
			var got []string
			for _, r := range Run(in) {
				if r.Rule == "complexity::cognitive_overflow" {
					got = append(got, r.Message)
				}
			}
			want := []string{fmt.Sprintf("function f2 is too complex: %s %d exceeds %d", tc.label, tc.limit+1, tc.limit)}
			if !slices.Equal(got, want) {
				t.Errorf("%s limit %s=%d (%s): messages %q, want %q", tc.measure, tc.key, tc.limit, name, got, want)
			}
		}
	}
}

// TestCognitiveOverflowDefaults checks the default limits of SPEC Sec 10
// against the metric values: 3 levels, 4 parameters, 60 lines, score 15.
func TestCognitiveOverflowDefaults(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for measure, limit := range map[string]int{"params": 4, "lines": 60, "depth": 3, "score": 15} {
		src := thresholdSrc(measure, limit)
		in := check(t, KindGo, "x.go", src)
		var got []string
		for _, r := range Run(in) {
			got = append(got, fmt.Sprintf("%d %s %s", r.Pos.Line(), r.Rule, r.Severity))
		}
		if len(got) != 1 || !strings.HasSuffix(got[0], " complexity::cognitive_overflow deny") {
			t.Errorf("%s: results %q, want one deny finding for f2", measure, got)
		}
		if !strings.Contains(src, "func f2") {
			t.Fatalf("%s: no f2 in source", measure)
		}
	}
}

// TestCognitiveOverflowReport checks the message of a function over several
// limits, the span, and the directive.
func TestCognitiveOverflowReport(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func work() {}

func many(a, b, c, d, e int) {
	for a > 0 {
		if b > 0 {
			for c > 0 {
				if d > 0 {
					work()
				}
			}
		}
	}
}

//logos:allow(complexity::cognitive_overflow)
func allowed(a, b, c, d, e int) {}

func literal() {
	f := func(a, b, c, d, e int) {}
	f(1, 2, 3, 4, 5)
}
`
	rs := Run(check(t, KindGo, "x.go", src))
	var got []string
	for _, r := range rs {
		got = append(got, fmt.Sprintf("%d %s", r.Pos.Line(), r.Message))
	}
	want := []string{
		"5 function many is too complex: nesting depth 4 exceeds 3, parameter count 5 exceeds 4",
		"21 function literal is too complex: parameter count 5 exceeds 4",
	}
	if !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
	if len(rs) == 2 {
		if got := spanText(src, rs[1]); got != "func(a, b, c, d, e int) {}" {
			t.Errorf("literal span = %q", got)
		}
		if rs[0].Pos.Col() != 6 || rs[0].End.Line() != 15 {
			t.Errorf("declaration span = %v to %v, want the name at column 6 through line 15", rs[0].Pos, rs[0].End)
		}
	}
}

// TestCognitiveOverflowNested checks that a nested function body counts
// toward the lines of its enclosing function and nothing else.
func TestCognitiveOverflowNested(t *testing.T) {
	src := "package p\n\nfunc work() {}\n\nfunc outer() {\n\tf := func() {\n" +
		strings.Repeat("\t\tif true {\n\t\t}\n", 3) + "\t}\n\tf()\n}\n"
	got := measureFuncs(t, KindGo, "x.go", src)
	compareFuncMetrics(t, "x.go", got, map[string]wantMetrics{
		"work":     {0, 0, 1, 0},
		"outer":    {0, 0, 11, 0},
		"literal1": {1, 0, 8, 3},
	})
}

// TestCognitiveOverflowNoBody checks that a declaration without a body is
// not a function to measure.
func TestCognitiveOverflowNoBody(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindGo, "x.go", "package p\n\nfunc f(a, b, c, d, e int)\n")
	if got := summarize(Run(in)); len(got) != 0 {
		t.Errorf("results = %q, want none", got)
	}
}
