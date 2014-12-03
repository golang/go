// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"appengine"
	"appengine/datastore"
)

func init() {
	handleFunc("/perfdetail", perfDetailUIHandler)
}

func perfDetailUIHandler(w http.ResponseWriter, r *http.Request) {
	d := dashboardForRequest(r)
	c := d.Context(appengine.NewContext(r))
	pc, err := GetPerfConfig(c, r)
	if err != nil {
		logErr(w, r, err)
		return
	}

	kind := r.FormValue("kind")
	builder := r.FormValue("builder")
	benchmark := r.FormValue("benchmark")
	if kind == "" {
		kind = "benchmark"
	}
	if kind != "benchmark" && kind != "builder" {
		logErr(w, r, fmt.Errorf("unknown kind %s", kind))
		return
	}

	// Fetch the new commit.
	com1 := new(Commit)
	com1.Hash = r.FormValue("commit")
	if hash, ok := knownTags[com1.Hash]; ok {
		com1.Hash = hash
	}
	if err := datastore.Get(c, com1.Key(c), com1); err != nil {
		logErr(w, r, fmt.Errorf("failed to fetch commit %s: %v", com1.Hash, err))
		return
	}
	// Fetch the associated perf result.
	ress1 := &PerfResult{CommitHash: com1.Hash}
	if err := datastore.Get(c, ress1.Key(c), ress1); err != nil {
		logErr(w, r, fmt.Errorf("failed to fetch perf result %s: %v", com1.Hash, err))
		return
	}

	// Fetch the old commit.
	var ress0 *PerfResult
	com0 := new(Commit)
	com0.Hash = r.FormValue("commit0")
	if hash, ok := knownTags[com0.Hash]; ok {
		com0.Hash = hash
	}
	if com0.Hash != "" {
		// Have an exact commit hash, fetch directly.
		if err := datastore.Get(c, com0.Key(c), com0); err != nil {
			logErr(w, r, fmt.Errorf("failed to fetch commit %s: %v", com0.Hash, err))
			return
		}
		ress0 = &PerfResult{CommitHash: com0.Hash}
		if err := datastore.Get(c, ress0.Key(c), ress0); err != nil {
			logErr(w, r, fmt.Errorf("failed to fetch perf result for %s: %v", com0.Hash, err))
			return
		}
	} else {
		// Don't have the commit hash, find the previous commit to compare.
		rc := MakePerfResultCache(c, com1, false)
		ress0, err = rc.NextForComparison(com1.Num, "")
		if err != nil {
			logErr(w, r, err)
			return
		}
		if ress0 == nil {
			logErr(w, r, fmt.Errorf("no previous commit with results"))
			return
		}
		// Now that we know the right result, fetch the commit.
		com0.Hash = ress0.CommitHash
		if err := datastore.Get(c, com0.Key(c), com0); err != nil {
			logErr(w, r, fmt.Errorf("failed to fetch commit %s: %v", com0.Hash, err))
			return
		}
	}

	res0 := ress0.ParseData()
	res1 := ress1.ParseData()
	var benchmarks []*uiPerfDetailBenchmark
	var list []string
	if kind == "builder" {
		list = pc.BenchmarksForBuilder(builder)
	} else {
		list = pc.BuildersForBenchmark(benchmark)
	}
	for _, other := range list {
		if kind == "builder" {
			benchmark = other
		} else {
			builder = other
		}
		var procs []*uiPerfDetailProcs
		allProcs := pc.ProcList(builder)
		for _, p := range allProcs {
			BenchProcs := fmt.Sprintf("%v-%v", benchmark, p)
			if res0[builder] == nil || res0[builder][BenchProcs] == nil {
				continue
			}
			pp := &uiPerfDetailProcs{Procs: p}
			for metric, val := range res0[builder][BenchProcs].Metrics {
				var pm uiPerfDetailMetric
				pm.Name = metric
				pm.Val0 = fmt.Sprintf("%v", val)
				val1 := uint64(0)
				if res1[builder] != nil && res1[builder][BenchProcs] != nil {
					val1 = res1[builder][BenchProcs].Metrics[metric]
				}
				pm.Val1 = fmt.Sprintf("%v", val1)
				v0 := val
				v1 := val1
				valf := perfDiff(v0, v1)
				pm.Delta = fmt.Sprintf("%+.2f%%", valf)
				pm.Style = perfChangeStyle(pc, valf, builder, BenchProcs, pm.Name)
				pp.Metrics = append(pp.Metrics, pm)
			}
			sort.Sort(pp.Metrics)
			for artifact, hash := range res0[builder][BenchProcs].Artifacts {
				var pm uiPerfDetailMetric
				pm.Val0 = fmt.Sprintf("%v", artifact)
				pm.Link0 = fmt.Sprintf("log/%v", hash)
				pm.Val1 = fmt.Sprintf("%v", artifact)
				if res1[builder] != nil && res1[builder][BenchProcs] != nil && res1[builder][BenchProcs].Artifacts[artifact] != "" {
					pm.Link1 = fmt.Sprintf("log/%v", res1[builder][BenchProcs].Artifacts[artifact])
				}
				pp.Metrics = append(pp.Metrics, pm)
			}
			procs = append(procs, pp)
		}
		benchmarks = append(benchmarks, &uiPerfDetailBenchmark{other, procs})
	}

	cfg := new(uiPerfConfig)
	for _, v := range pc.BuildersForBenchmark("") {
		cfg.Builders = append(cfg.Builders, uiPerfConfigElem{v, v == builder})
	}
	for _, v := range pc.BenchmarksForBuilder("") {
		cfg.Benchmarks = append(cfg.Benchmarks, uiPerfConfigElem{v, v == benchmark})
	}

	data := &uiPerfDetailTemplateData{d, cfg, kind == "builder", com0, com1, benchmarks}

	var buf bytes.Buffer
	if err := uiPerfDetailTemplate.Execute(&buf, data); err != nil {
		logErr(w, r, err)
		return
	}

	buf.WriteTo(w)
}

func perfResultSplit(s string) (builder string, benchmark string, procs int) {
	s1 := strings.Split(s, "|")
	s2 := strings.Split(s1[1], "-")
	procs, _ = strconv.Atoi(s2[1])
	return s1[0], s2[0], procs
}

type uiPerfDetailTemplateData struct {
	Dashboard   *Dashboard
	Config      *uiPerfConfig
	KindBuilder bool
	Commit0     *Commit
	Commit1     *Commit
	Benchmarks  []*uiPerfDetailBenchmark
}

type uiPerfDetailBenchmark struct {
	Name  string
	Procs []*uiPerfDetailProcs
}

type uiPerfDetailProcs struct {
	Procs   int
	Metrics uiPerfDetailMetrics
}

type uiPerfDetailMetric struct {
	Name  string
	Val0  string
	Val1  string
	Link0 string
	Link1 string
	Delta string
	Style string
}

type uiPerfDetailMetrics []uiPerfDetailMetric

func (l uiPerfDetailMetrics) Len() int           { return len(l) }
func (l uiPerfDetailMetrics) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }
func (l uiPerfDetailMetrics) Less(i, j int) bool { return l[i].Name < l[j].Name }

var uiPerfDetailTemplate = template.Must(
	template.New("perf_detail.html").Funcs(tmplFuncs).ParseFiles("build/perf_detail.html"),
)
