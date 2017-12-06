// sample program that is used to produce some of the files in
// pprof/internal/report/testdata.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime/pprof"
)

var cpuProfile = flag.String("cpuprofile", "", "where to write cpu profile")

func main() {
	flag.Parse()
	f, err := os.Create(*cpuProfile)
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()
	busyLoop()
}

func busyLoop() {
	m := make(map[int]int)
	for i := 0; i < 1000000; i++ {
		m[i] = i + 10
	}
	var sum float64
	for i := 0; i < 100; i++ {
		for _, v := range m {
			sum += math.Abs(float64(v))
		}
	}
	fmt.Println("Sum", sum)
}
