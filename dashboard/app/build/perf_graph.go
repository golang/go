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
	"strconv"

	"appengine"
	"appengine/datastore"
)

func init() {
	for _, d := range dashboards {
		http.HandleFunc(d.RelPath+"perfgraph", perfGraphHandler)
	}
}

func perfGraphHandler(w http.ResponseWriter, r *http.Request) {
	d := dashboardForRequest(r)
	c := d.Context(appengine.NewContext(r))
	pc, err := GetPerfConfig(c, r)
	if err != nil {
		logErr(w, r, err)
		return
	}
	allBuilders := pc.BuildersForBenchmark("")
	allBenchmarks := pc.BenchmarksForBuilder("")
	allMetrics := pc.MetricsForBenchmark("")
	allProcs := pc.ProcList("")
	r.ParseForm()
	selBuilders := r.Form["builder"]
	selBenchmarks := r.Form["benchmark"]
	selMetrics := r.Form["metric"]
	selProcs := r.Form["procs"]
	if len(selBuilders) == 0 {
		selBuilders = append(selBuilders, allBuilders[0])
	}
	if len(selBenchmarks) == 0 {
		selBenchmarks = append(selBenchmarks, "json")
	}
	if len(selMetrics) == 0 {
		selMetrics = append(selMetrics, "time")
	}
	if len(selProcs) == 0 {
		selProcs = append(selProcs, "1")
	}
	commitFrom := r.FormValue("commit-from")
	if commitFrom == "" {
		commitFrom = lastRelease
	}
	commitTo := r.FormValue("commit-to")
	if commitTo == "" {
		commitTo = "tip"
	}
	// TODO(dvyukov): validate input

	// Figure out start and end commit from commitFrom/commitTo.
	startCommitNum := 0
	endCommitNum := 0
	{
		comFrom := &Commit{Hash: knownTags[commitFrom]}
		if err := datastore.Get(c, comFrom.Key(c), comFrom); err != nil {
			logErr(w, r, err)
			return
		}
		startCommitNum = comFrom.Num - 1

	retry:
		if commitTo == "tip" {
			p, err := GetPackage(c, "")
			if err != nil {
				logErr(w, r, err)
				return
			}
			endCommitNum = p.NextNum
		} else {
			comTo := &Commit{Hash: knownTags[commitTo]}
			if err := datastore.Get(c, comTo.Key(c), comTo); err != nil {
				logErr(w, r, err)
				return
			}
			endCommitNum = comTo.Num
		}
		if endCommitNum <= startCommitNum {
			// User probably selected from:go1.3 to:go1.2. Fix go1.2 to tip.
			if commitTo == "tip" {
				logErr(w, r, fmt.Errorf("no commits to display (%v-%v)", commitFrom, commitTo))
				return
			}
			commitTo = "tip"
			goto retry
		}
	}
	commitsToDisplay := endCommitNum - startCommitNum

	present := func(set []string, s string) bool {
		for _, s1 := range set {
			if s1 == s {
				return true
			}
		}
		return false
	}

	cfg := &uiPerfConfig{}
	for _, v := range allBuilders {
		cfg.Builders = append(cfg.Builders, uiPerfConfigElem{v, present(selBuilders, v)})
	}
	for _, v := range allBenchmarks {
		cfg.Benchmarks = append(cfg.Benchmarks, uiPerfConfigElem{v, present(selBenchmarks, v)})
	}
	for _, v := range allMetrics {
		cfg.Metrics = append(cfg.Metrics, uiPerfConfigElem{v, present(selMetrics, v)})
	}
	for _, v := range allProcs {
		cfg.Procs = append(cfg.Procs, uiPerfConfigElem{strconv.Itoa(v), present(selProcs, strconv.Itoa(v))})
	}
	for k := range knownTags {
		cfg.CommitsFrom = append(cfg.CommitsFrom, uiPerfConfigElem{k, commitFrom == k})
	}
	for k := range knownTags {
		cfg.CommitsTo = append(cfg.CommitsTo, uiPerfConfigElem{k, commitTo == k})
	}
	cfg.CommitsTo = append(cfg.CommitsTo, uiPerfConfigElem{"tip", commitTo == "tip"})

	var vals [][]float64
	var hints [][]string
	var certainty [][]bool
	var headers []string
	commits2, err := GetCommits(c, startCommitNum, commitsToDisplay)
	if err != nil {
		logErr(w, r, err)
		return
	}
	for _, builder := range selBuilders {
		for _, metric := range selMetrics {
			for _, benchmark := range selBenchmarks {
				for _, procs := range selProcs {
					benchProcs := fmt.Sprintf("%v-%v", benchmark, procs)
					vv, err := GetPerfMetricsForCommits(c, builder, benchProcs, metric, startCommitNum, commitsToDisplay)
					if err != nil {
						logErr(w, r, err)
						return
					}
					hasdata := false
					for _, v := range vv {
						if v != 0 {
							hasdata = true
						}
					}
					if hasdata {
						descBuilder := "/" + builder
						descBenchmark := "/" + benchProcs
						descMetric := "/" + metric
						if len(selBuilders) == 1 {
							descBuilder = ""
						}
						if len(selBenchmarks) == 1 && len(selProcs) == 1 {
							descBenchmark = ""
						}
						if len(selMetrics) == 1 && (len(selBuilders) > 1 || len(selBenchmarks) > 1 || len(selProcs) > 1) {
							descMetric = ""
						}
						desc := fmt.Sprintf("%v%v%v", descBuilder, descBenchmark, descMetric)[1:]
						hh := make([]string, commitsToDisplay)
						valf := make([]float64, commitsToDisplay)
						cert := make([]bool, commitsToDisplay)
						firstval := uint64(0)
						lastval := uint64(0)
						lastval0 := uint64(0)
						for i, v := range vv {
							cert[i] = true
							if v == 0 {
								if lastval == 0 {
									continue
								}
								nextval := uint64(0)
								nextidx := 0
								for i2, v2 := range vv[i+1:] {
									if v2 != 0 {
										nextval = v2
										nextidx = i + i2 + 1
										break
									}
								}
								if nextval == 0 {
									continue
								}
								cert[i] = false
								v = lastval + uint64(int64(nextval-lastval)/int64(nextidx-i+1))
							}
							if firstval == 0 {
								firstval = v
							}
							valf[i] = float64(v) / float64(firstval)
							if cert[i] {
								d := ""
								if lastval0 != 0 {
									d = fmt.Sprintf(" (%.02f%%)", perfDiff(lastval0, v))
								}
								hh[i] = fmt.Sprintf("%v%v", v, d)

							} else {
								hh[i] = "NO DATA"
							}
							lastval = v
							if cert[i] {
								lastval0 = v
							}
						}
						vals = append(vals, valf)
						hints = append(hints, hh)
						certainty = append(certainty, cert)
						headers = append(headers, desc)
					}
				}
			}
		}
	}

	var commits []perfGraphCommit
	if len(vals) != 0 && len(vals[0]) != 0 {
		idx := 0
		for i := range vals[0] {
			com := commits2[i]
			if com == nil || !com.NeedsBenchmarking {
				continue
			}
			c := perfGraphCommit{Id: idx, Name: fmt.Sprintf("%v (%v)", com.Desc, com.Time.Format("Jan 2, 2006 1:04"))}
			idx++
			for j := range vals {
				c.Vals = append(c.Vals, perfGraphValue{float64(vals[j][i]), certainty[j][i], hints[j][i]})
			}
			commits = append(commits, c)
		}
	}

	data := &perfGraphData{d, cfg, headers, commits}

	var buf bytes.Buffer
	if err := perfGraphTemplate.Execute(&buf, data); err != nil {
		logErr(w, r, err)
		return
	}

	buf.WriteTo(w)
}

var perfGraphTemplate = template.Must(
	template.New("perf_graph.html").ParseFiles("build/perf_graph.html"),
)

type perfGraphData struct {
	Dashboard *Dashboard
	Config    *uiPerfConfig
	Headers   []string
	Commits   []perfGraphCommit
}

type perfGraphCommit struct {
	Id   int
	Name string
	Vals []perfGraphValue
}

type perfGraphValue struct {
	Val       float64
	Certainty bool
	Hint      string
}
