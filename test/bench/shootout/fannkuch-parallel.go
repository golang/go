/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * Based on fannkuch.scala by Rex Kerr
 */

package main

import (
	"flag"
	"fmt"
	"runtime"
)

var n = flag.Int("n", 7, "count")
var nCPU = flag.Int("ncpu", 4, "number of cpus")

type Job struct {
	start []int
	n     int
}

type Found struct {
	who *Kucher
	k   int
}

type Kucher struct {
	perm []int
	temp []int
	flip []int
	in   chan Job
}

func NewKucher(length int) *Kucher {
	return &Kucher{
		perm: make([]int, length),
		temp: make([]int, length),
		flip: make([]int, length),
		in:   make(chan Job),
	}
}

func (k *Kucher) permute(n int) bool {
	i := 0
	for ; i < n-1 && k.flip[i] == 0; i++ {
		t := k.perm[0]
		j := 0
		for ; j <= i; j++ {
			k.perm[j] = k.perm[j+1]
		}
		k.perm[j] = t
	}
	k.flip[i]--
	for i > 0 {
		i--
		k.flip[i] = i
	}
	return k.flip[n-1] >= 0
}

func (k *Kucher) count() int {
	K := 0
	copy(k.temp, k.perm)
	for k.temp[0] != 0 {
		m := k.temp[0]
		for i := 0; i < m; i++ {
			k.temp[i], k.temp[m] = k.temp[m], k.temp[i]
			m--
		}
		K++
	}
	return K
}

func (k *Kucher) Run(foreman chan<- Found) {
	for job := range k.in {
		verbose := 30
		copy(k.perm, job.start)
		for i, v := range k.perm {
			if v != i {
				verbose = 0
			}
			k.flip[i] = i
		}
		K := 0
		for {
			if verbose > 0 {
				for _, p := range k.perm {
					fmt.Print(p + 1)
				}
				fmt.Println()
				verbose--
			}
			count := k.count()
			if count > K {
				K = count
			}
			if !k.permute(job.n) {
				break
			}
		}
		foreman <- Found{k, K}
	}
}

type Fanner struct {
	jobind   int
	jobsdone int
	k        int
	jobs     []Job
	workers  []*Kucher
	in       chan Found
	result   chan int
}

func NewFanner(jobs []Job, workers []*Kucher) *Fanner {
	return &Fanner{
		jobs: jobs, workers: workers,
		in:     make(chan Found),
		result: make(chan int),
	}
}

func (f *Fanner) Run(N int) {
	for msg := range f.in {
		if msg.k > f.k {
			f.k = msg.k
		}
		if msg.k >= 0 {
			f.jobsdone++
		}
		if f.jobind < len(f.jobs) {
			msg.who.in <- f.jobs[f.jobind]
			f.jobind++
		} else if f.jobsdone == len(f.jobs) {
			f.result <- f.k
			return
		}
	}
}

func swapped(a []int, i, j int) []int {
	b := make([]int, len(a))
	copy(b, a)
	b[i], b[j] = a[j], a[i]
	return b
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(*nCPU)
	N := *n
	base := make([]int, N)
	for i := range base {
		base[i] = i
	}

	njobs := 1
	if N > 8 {
		njobs += (N*(N-1))/2 - 28 // njobs = 1 + sum(8..N-1) = 1 + sum(1..N-1) - sum(1..7)
	}
	jobs := make([]Job, njobs)
	jobsind := 0

	firstN := N
	if firstN > 8 {
		firstN = 8
	}
	jobs[jobsind] = Job{base, firstN}
	jobsind++
	for i := N - 1; i >= 8; i-- {
		for j := 0; j < i; j++ {
			jobs[jobsind] = Job{swapped(base, i, j), i}
			jobsind++
		}
	}

	nworkers := *nCPU
	if njobs < nworkers {
		nworkers = njobs
	}
	workers := make([]*Kucher, nworkers)
	foreman := NewFanner(jobs, workers)
	go foreman.Run(N)
	for i := range workers {
		k := NewKucher(N)
		workers[i] = k
		go k.Run(foreman.in)
		foreman.in <- Found{k, -1}
	}
	fmt.Printf("Pfannkuchen(%d) = %d\n", N, <-foreman.result)
}
