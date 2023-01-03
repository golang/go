// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package callgraph_test

import (
	"log"
	"sync"
	"testing"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/callgraph/static"
	"golang.org/x/tools/go/callgraph/vta"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Benchmarks comparing different callgraph algorithms implemented in
// x/tools/go/callgraph. Comparison is on both speed, memory and precision.
// Fewer edges and fewer reachable nodes implies a more precise result.
// Comparison is done on a hello world http server using net/http.
//
// Current results were on an i7 macbook on go version devel go1.20-2730.
// Number of nodes, edges, and reachable function are expected to vary between
// go versions. Timing results are expected to vary between machines.
//
// BenchmarkStatic-12	 53 ms/op     6 MB/op	12113 nodes	 37355 edges	1522 reachable
// BenchmarkCHA-12    	 86 ms/op	 16 MB/op	12113 nodes	131717 edges	7640 reachable
// BenchmarkRTA-12		110 ms/op	 12 MB/op	 6566 nodes	 42291 edges	5099 reachable
// BenchmarkPTA-12	   1427 ms/op	600 MB/op	 8714 nodes	 28244 edges	4184 reachable
// BenchmarkVTA-12		603 ms/op	 87 MB/op	12112 nodes	 44857 edges	4918 reachable
// BenchmarkVTA2-12		774 ms/op	115 MB/op	 4918 nodes	 20247 edges	3841 reachable
// BenchmarkVTA3-12		928 ms/op	134 MB/op	 3841 nodes	 16502 edges	3542 reachable
// BenchmarkVTAAlt-12	395 ms/op	 61 MB/op	 7640 nodes  29629 edges	4257 reachable
// BenchmarkVTAAlt2-12	556 ms/op	 82 MB/op	 4257 nodes	 18057 edges	3586 reachable
//
// Note:
// * Static is unsound and may miss real edges.
// * RTA starts from a main function and only includes reachable functions.
// * CHA starts from all functions.
// * VTA, VTA2, and VTA3 are starting from all functions and the CHA callgraph.
//   VTA2 and VTA3 are the result of re-applying VTA to the functions reachable
//   from main() via the callgraph of the previous stage.
// * VTAAlt, and VTAAlt2 start from the functions reachable from main via the
//   CHA callgraph.
// * All algorithms are unsound w.r.t. reflection.

const httpEx = `package main

import (
    "fmt"
    "net/http"
)

func hello(w http.ResponseWriter, req *http.Request) {
    fmt.Fprintf(w, "hello world\n")
}

func main() {
    http.HandleFunc("/hello", hello)
    http.ListenAndServe(":8090", nil)
}
`

var (
	once sync.Once
	prog *ssa.Program
	main *ssa.Function
)

func example() (*ssa.Program, *ssa.Function) {
	once.Do(func() {
		var conf loader.Config
		f, err := conf.ParseFile("<input>", httpEx)
		if err != nil {
			log.Fatal(err)
		}
		conf.CreateFromFiles(f.Name.Name, f)

		lprog, err := conf.Load()
		if err != nil {
			log.Fatalf("test 'package %s': Load: %s", f.Name.Name, err)
		}
		prog = ssautil.CreateProgram(lprog, ssa.InstantiateGenerics)
		prog.Build()

		main = prog.Package(lprog.Created[0].Pkg).Members["main"].(*ssa.Function)
	})
	return prog, main
}

var stats bool = false // print stats?

func logStats(b *testing.B, name string, cg *callgraph.Graph, main *ssa.Function) {
	if stats {
		e := 0
		for _, n := range cg.Nodes {
			e += len(n.Out)
		}
		r := len(reaches(cg, main))
		b.Logf("%s:\t%d nodes\t%d edges\t%d reachable", name, len(cg.Nodes), e, r)
	}
}

func BenchmarkStatic(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		cg := static.CallGraph(prog)
		logStats(b, "static", cg, main)
	}
}

func BenchmarkCHA(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		cg := cha.CallGraph(prog)
		logStats(b, "cha", cg, main)
	}
}

func BenchmarkRTA(b *testing.B) {
	b.StopTimer()
	_, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		res := rta.Analyze([]*ssa.Function{main}, true)
		cg := res.CallGraph
		logStats(b, "rta", cg, main)
	}
}

func BenchmarkPTA(b *testing.B) {
	b.StopTimer()
	_, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		config := &pointer.Config{Mains: []*ssa.Package{main.Pkg}, BuildCallGraph: true}
		res, err := pointer.Analyze(config)
		if err != nil {
			b.Fatal(err)
		}
		logStats(b, "pta", res.CallGraph, main)
	}
}

func BenchmarkVTA(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		cg := vta.CallGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
		logStats(b, "vta", cg, main)
	}
}

func BenchmarkVTA2(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		vta1 := vta.CallGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
		cg := vta.CallGraph(reaches(vta1, main), vta1)
		logStats(b, "vta2", cg, main)
	}
}

func BenchmarkVTA3(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		vta1 := vta.CallGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
		vta2 := vta.CallGraph(reaches(vta1, main), vta1)
		cg := vta.CallGraph(reaches(vta2, main), vta2)
		logStats(b, "vta3", cg, main)
	}
}

func BenchmarkVTAAlt(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		cha := cha.CallGraph(prog)
		cg := vta.CallGraph(reaches(cha, main), cha) // start from only functions reachable by CHA.
		logStats(b, "vta-alt", cg, main)
	}
}

func BenchmarkVTAAlt2(b *testing.B) {
	b.StopTimer()
	prog, main := example()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		cha := cha.CallGraph(prog)
		vta1 := vta.CallGraph(reaches(cha, main), cha)
		cg := vta.CallGraph(reaches(vta1, main), vta1)
		logStats(b, "vta-alt2", cg, main)
	}
}

// reaches returns the set of functions forward reachable from f in g.
func reaches(g *callgraph.Graph, f *ssa.Function) map[*ssa.Function]bool {
	seen := make(map[*ssa.Function]bool)
	var visit func(n *callgraph.Node)
	visit = func(n *callgraph.Node) {
		if !seen[n.Func] {
			seen[n.Func] = true
			for _, e := range n.Out {
				visit(e.Callee)
			}
		}
	}
	visit(g.Nodes[f])
	return seen
}
