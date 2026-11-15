// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

const rounds = 3

func mkSeed(idx, which int) uint64 {
	return uint64(idx*100 + which*10)
}

func mkP(n int, seed uint64) []byte {
	b := make([]byte, n)
	fillB(b, seed)
	return b
}

func fillB(b []byte, seed uint64) {
	for i := range b {
		b[i] = byte(seed*131+uint64(i))%251 + 1
	}
}

func checksum3(a, b, c []byte) uint64 {
	var h uint64 = 14695981039346656037
	for _, buf := range [3][]byte{a, b, c} {
		for i := range buf {
			h ^= uint64(buf[i])
			h *= 1099511628211
		}
		h ^= uint64(len(buf))
		h *= 1099511628211
	}
	return h
}

func worker(idx, gw int, ping chan int, ack chan struct{}, out *string) {
	n1 := 64 + idx*2
	n2 := 1024 + idx*3
	n3 := 61440 - idx*16
	s1, s2, s3 := mkSeed(idx, 1), mkSeed(idx, 2), mkSeed(idx, 3)
	clobbered := false
	for round := range rounds {
		b1 := mkP(n1, s1)
		b2 := mkP(n2, s2)
		b3 := mkP(n3, s3)
		expect := checksum3(b1, b2, b3)
		ping <- idx
		<-ack
		if !clobbered && checksum3(b1, b2, b3) != expect {
			*out = fmt.Sprintf("LAW gw%d g%d round %d clobbered", gw, idx, round)
			clobbered = true
		}
	}
	if clobbered {
		return
	}
	*out = fmt.Sprintf("gw%d g%d ok", gw, idx)
}

func run(gw int) []string {
	ping := make([]chan int, gw)
	ack := make([]chan struct{}, gw)
	res := make([]string, gw)
	for i := range ping {
		ping[i] = make(chan int)
		ack[i] = make(chan struct{})
	}
	var wg sync.WaitGroup
	for i := range gw {
		wg.Go(func() {
			worker(i, gw, ping[i], ack[i], &res[i])
		})
	}
	for range rounds {
		for i := range gw {
			<-ping[i]
			ack[i] <- struct{}{}
		}
	}
	wg.Wait()
	return res
}

func main() {
	bad := 0
	for _, gw := range []int{2, 8, 64} {
		for _, line := range run(gw) {
			if strings.HasPrefix(line, "LAW ") {
				bad++
			}
			fmt.Println(line)
		}
	}
	if bad > 0 {
		fmt.Printf("LAW arena_park %d workers clobbered\n", bad)
		os.Exit(1)
	}
	fmt.Println("arena_park ok")
}
