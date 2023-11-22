// runoutput -goexperiment rangefunc

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Torture test for range-over-func.
//
// cmd/internal/testdir runs this like
//
//	go run rangegen.go >x.go
//	go run x.go
//
// but a longer version can be run using
//
//	go run rangegen.go long
//
// In that second form, rangegen takes care of compiling
// and running the code it generates, in batches.
// That form takes 10-20 minutes to run.

package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
)

const verbose = false

func main() {
	long := len(os.Args) > 1 && os.Args[1] == "long"
	log.SetFlags(0)
	log.SetPrefix("rangegen: ")

	b := new(bytes.Buffer)
	tests := ""
	flush := func(force bool) {
		if !long || (strings.Count(tests, "\n") < 1000 && !force) {
			return
		}
		p(b, mainCode, tests)
		err := os.WriteFile("tmp.go", b.Bytes(), 0666)
		if err != nil {
			log.Fatal(err)
		}
		out, err := exec.Command("go", "run", "tmp.go").CombinedOutput()
		if err != nil {
			log.Fatalf("go run tmp.go: %v\n%s", err, out)
		}
		print(".")
		if force {
			print("\nPASS\n")
		}
		b.Reset()
		tests = ""
		p(b, "package main\n\n")
		p(b, "const verbose = %v\n\n", verbose)
	}

	p(b, "package main\n\n")
	p(b, "const verbose = %v\n\n", verbose)
	max := 2
	if !long {
		max = 5
	}
	for i := 1; i <= max; i++ {
		maxDouble := -1
		if long {
			maxDouble = i
		}
		for double := -1; double <= maxDouble; double++ {
			code := gen(new(bytes.Buffer), "", "", "", i, double, func(c int) bool { return true })
			for j := 0; j < code; j++ {
				hi := j + 1
				if long {
					hi = code
				}
				for k := j; k < hi && k < code; k++ {
					s := fmt.Sprintf("%d_%d_%d_%d", i, double+1, j, k)
					code0 := gen(b, "testFunc"+s, "", "yield2", i, double, func(c int) bool { return c == j || c == k })
					code1 := gen(b, "testSlice"+s, "_, ", "slice2", i, double, func(c int) bool { return c == j || c == k })
					if code0 != code1 {
						panic("bad generator")
					}
					tests += "test" + s + "()\n"
					p(b, testCode, "test"+s, []int{j, k}, "testFunc"+s, "testSlice"+s)
					flush(false)
				}
			}
		}
	}
	for i := 1; i <= max; i++ {
		maxDouble := -1
		if long {
			maxDouble = i
		}
		for double := -1; double <= maxDouble; double++ {
			s := fmt.Sprintf("%d_%d", i, double+1)
			code := gen(b, "testFunc"+s, "", "yield2", i, double, func(c int) bool { return true })
			code1 := gen(b, "testSlice"+s, "_, ", "slice2", i, double, func(c int) bool { return true })
			if code != code1 {
				panic("bad generator")
			}
			tests += "test" + s + "()\n"
			var all []int
			for j := 0; j < code; j++ {
				all = append(all, j)
			}
			p(b, testCode, "test"+s, all, "testFunc"+s, "testSlice"+s)
			flush(false)
		}
	}
	if long {
		flush(true)
		os.Remove("tmp.go")
		return
	}

	p(b, mainCode, tests)

	os.Stdout.Write(b.Bytes())
}

func p(b *bytes.Buffer, format string, args ...any) {
	fmt.Fprintf(b, format, args...)
}

func gen(b *bytes.Buffer, name, prefix, rangeExpr string, depth, double int, allowed func(int) bool) int {
	p(b, "func %s(o *output, code int) int {\n", name)
	p(b, "	dfr := 0; _ = dfr\n")
	code := genLoop(b, 0, prefix, rangeExpr, depth, double, 0, "", allowed)
	p(b, "	return 0\n")
	p(b, "}\n\n")
	return code
}

func genLoop(b *bytes.Buffer, d int, prefix, rangeExpr string, depth, double, code int, labelSuffix string, allowed func(int) bool) int {
	limit := 1
	if d == double {
		limit = 2
	}
	for rep := 0; rep < limit; rep++ {
		if rep == 1 {
			labelSuffix = "R"
		}
		s := fmt.Sprintf("%d%s", d, labelSuffix)
		p(b, "	o.log(`top%s`)\n", s)
		p(b, "	l%sa := 0\n", s)
		p(b, "goto L%sa; L%sa:	o.log(`L%sa`)\n", s, s, s)
		p(b, "	if l%sa++; l%sa >= 2 { o.log(`loop L%sa`); return -1 }\n", s, s, s)
		p(b, "	l%sfor := 0\n", s)
		p(b, "goto L%sfor; L%sfor: for f := 0; f < 1; f++ { o.log(`L%sfor`)\n", s, s, s)
		p(b, "	if l%sfor++; l%sfor >= 2 { o.log(`loop L%sfor`); return -1 }\n", s, s, s)
		p(b, "	l%ssw := 0\n", s)
		p(b, "goto L%ssw; L%ssw: switch { default: o.log(`L%ssw`)\n", s, s, s)
		p(b, "	if l%ssw++; l%ssw >= 2 { o.log(`loop L%ssw`); return -1 }\n", s, s, s)
		p(b, "	l%ssel := 0\n", s)
		p(b, "goto L%ssel; L%ssel: select { default: o.log(`L%ssel`)\n", s, s, s)
		p(b, "	if l%ssel++; l%ssel >= 2 { o.log(`loop L%ssel`); return -1 }\n", s, s, s)
		p(b, "	l%s := 0\n", s)
		p(b, "goto L%s; L%s:	for %s i%s := range %s {\n", s, s, prefix, s, rangeExpr)
		p(b, "	o.log1(`L%s top`, i%s)\n", s, s)
		p(b, "	if l%s++; l%s >= 4 { o.log(`loop L%s`); return -1 }\n", s, s, s)
		printTests := func() {
			if code++; allowed(code) {
				p(b, "	if code == %v { break }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	if code == %v { continue }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	switch { case code == %v: continue }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	if code == %v { return %[1]v }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	if code == %v { select { default: break } }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	if code == %v { switch { default: break } }\n", code)
			}
			if code++; allowed(code) {
				p(b, "	if code == %v { dfr++; defer o.log1(`defer %d`, dfr) }\n", code, code)
			}
			for i := d; i > 0; i-- {
				suffix := labelSuffix
				if i < double {
					suffix = ""
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { break L%d%s }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { select { default: break L%d%s } }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { break L%d%s }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { break L%d%ssw }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { break L%d%ssel }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { break L%d%sfor }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { continue L%d%sfor }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { goto L%d%sa }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { goto L%d%s }\n", code, i, suffix)
				}
				if code++; allowed(code) {
					p(b, "	if code == %v { goto L%d%sb }\n", code, i, suffix)
				}
			}
		}
		printTests()
		if d < depth {
			if rep == 1 {
				double = d // signal to children to use the rep=1 labels
			}
			code = genLoop(b, d+1, prefix, rangeExpr, depth, double, code, labelSuffix, allowed)
			printTests()
		}
		p(b, "	o.log(`L%s bot`)\n", s)
		p(b, "	}\n")
		p(b, "	o.log(`L%ssel bot`)\n", s)
		p(b, "	}\n")
		p(b, "	o.log(`L%ssw bot`)\n", s)
		p(b, "	}\n")
		p(b, "	o.log(`L%sfor bot`)\n", s)
		p(b, "	}\n")
		p(b, "	o.log(`done%s`)\n", s)
		p(b, "goto L%sb; L%sb: o.log(`L%sb`)\n", s, s, s)
	}
	return code
}

var testCode = `
func %s() {
	all := %#v
	for i := 0; i < len(all); i++ {
		c := all[i]
		outFunc := run(%s, c)
		outSlice := run(%s, c)
		if !outFunc.eq(outSlice) {
			println("mismatch", "%[3]s", "%[4]s", c)
			println()
			println("func:")
			outFunc.print()
			println()
			println("slice:")
			outSlice.print()
			panic("mismatch")
		}
	}
	if verbose {
		println("did", "%[3]s", "%[4]s", len(all))
	}
}
`

var mainCode = `

func main() {
	if verbose {
		println("main")
	}
	%s
}

func yield2(yield func(int)bool) { _ = yield(1) && yield(2) }
var slice2 = []int{1,2}

type output struct {
	ret int
	trace []any
}

func (o *output) log(x any) {
	o.trace = append(o.trace, x)
}

func (o *output) log1(x, y any) {
	o.trace = append(o.trace, x, y)
}

func (o *output) eq(p *output) bool{
	if o.ret != p.ret  || len(o.trace) != len(p.trace) {
		return false
	}
	for i ,x := range o.trace {
		if x != p.trace[i] {
			return false
		}
	}
	return true
}

func (o *output) print() {
	println("ret", o.ret, "trace-len", len(o.trace))
	for i := 0; i < len(o.trace); i++ {
		print("#", i, " ")
		switch x := o.trace[i].(type) {
		case int:
			print(x)
		case string:
			print(x)
		default:
			print(x)
		}
		print("\n")
	}
}

func run(f func(*output, int)int, i int) *output {
	o := &output{}
	o.ret = f(o, i)
	return o
}

`
