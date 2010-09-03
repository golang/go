package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"http"
	"os"
	"regexp"
)

// getHighWater returns the current highwater revision hash for this builder
func (b *Builder) getHighWater() (rev string, err os.Error) {
	url := fmt.Sprintf("http://%s/hw-get?builder=%s",
		*dashboardhost, b.name)
	r, _, err := http.Get(url)
	if err != nil {
		return
	}
	buf := new(bytes.Buffer)
	_, err = buf.ReadFrom(r.Body)
	if err != nil {
		return
	}
	r.Body.Close()
	return buf.String(), nil
}

// recordResult sends build results to the dashboard
func (b *Builder) recordResult(buildLog string, c Commit) os.Error {
	return httpCommand("build", map[string]string{
		"builder": b.name,
		"key":     b.key,
		"node":    c.node,
		"parent":  c.parent,
		"user":    c.user,
		"date":    c.date,
		"desc":    c.desc,
		"log":     buildLog,
	})
}

// match lines like: "package.BechmarkFunc	100000	    999 ns/op"
var benchmarkRegexp = regexp.MustCompile("([^\n\t ]+)[\t ]+([0-9]+)[\t ]+([0-9]+) ns/op")

// recordBenchmarks sends benchmark results to the dashboard
func (b *Builder) recordBenchmarks(benchLog string, c Commit) os.Error {
	results := benchmarkRegexp.FindAllStringSubmatch(benchLog, -1)
	var buf bytes.Buffer
	b64 := base64.NewEncoder(base64.StdEncoding, &buf)
	for _, r := range results {
		for _, s := range r[1:] {
			binary.Write(b64, binary.BigEndian, uint16(len(s)))
			b64.Write([]byte(s))
		}
	}
	b64.Close()
	return httpCommand("benchmarks", map[string]string{
		"builder":       b.name,
		"key":           b.key,
		"node":          c.node,
		"benchmarkdata": buf.String(),
	})
}

func httpCommand(cmd string, args map[string]string) os.Error {
	url := fmt.Sprintf("http://%v/%v", *dashboardhost, cmd)
	_, err := http.PostForm(url, args)
	return err
}
