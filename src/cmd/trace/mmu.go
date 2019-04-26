// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Minimum mutator utilization (MMU) graphing.

// TODO:
//
// In worst window list, show break-down of GC utilization sources
// (STW, assist, etc). Probably requires a different MutatorUtil
// representation.
//
// When a window size is selected, show a second plot of the mutator
// utilization distribution for that window size.
//
// Render plot progressively so rough outline is visible quickly even
// for very complex MUTs. Start by computing just a few window sizes
// and then add more window sizes.
//
// Consider using sampling to compute an approximate MUT. This would
// work by sampling the mutator utilization at randomly selected
// points in time in the trace to build an empirical distribution. We
// could potentially put confidence intervals on these estimates and
// render this progressively as we refine the distributions.

package main

import (
	"encoding/json"
	"fmt"
	"internal/trace"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

func init() {
	http.HandleFunc("/mmu", httpMMU)
	http.HandleFunc("/mmuPlot", httpMMUPlot)
	http.HandleFunc("/mmuDetails", httpMMUDetails)
}

var utilFlagNames = map[string]trace.UtilFlags{
	"perProc":    trace.UtilPerProc,
	"stw":        trace.UtilSTW,
	"background": trace.UtilBackground,
	"assist":     trace.UtilAssist,
	"sweep":      trace.UtilSweep,
}

type mmuCacheEntry struct {
	init     sync.Once
	util     [][]trace.MutatorUtil
	mmuCurve *trace.MMUCurve
	err      error
}

var mmuCache struct {
	m    map[trace.UtilFlags]*mmuCacheEntry
	lock sync.Mutex
}

func init() {
	mmuCache.m = make(map[trace.UtilFlags]*mmuCacheEntry)
}

func getMMUCurve(r *http.Request) ([][]trace.MutatorUtil, *trace.MMUCurve, error) {
	var flags trace.UtilFlags
	for _, flagStr := range strings.Split(r.FormValue("flags"), "|") {
		flags |= utilFlagNames[flagStr]
	}

	mmuCache.lock.Lock()
	c := mmuCache.m[flags]
	if c == nil {
		c = new(mmuCacheEntry)
		mmuCache.m[flags] = c
	}
	mmuCache.lock.Unlock()

	c.init.Do(func() {
		events, err := parseEvents()
		if err != nil {
			c.err = err
		} else {
			c.util = trace.MutatorUtilization(events, flags)
			c.mmuCurve = trace.NewMMUCurve(c.util)
		}
	})
	return c.util, c.mmuCurve, c.err
}

// httpMMU serves the MMU plot page.
func httpMMU(w http.ResponseWriter, r *http.Request) {
	http.ServeContent(w, r, "", time.Time{}, strings.NewReader(templMMU))
}

// httpMMUPlot serves the JSON data for the MMU plot.
func httpMMUPlot(w http.ResponseWriter, r *http.Request) {
	mu, mmuCurve, err := getMMUCurve(r)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse events: %v", err), http.StatusInternalServerError)
		return
	}

	var quantiles []float64
	for _, flagStr := range strings.Split(r.FormValue("flags"), "|") {
		if flagStr == "mut" {
			quantiles = []float64{0, 1 - .999, 1 - .99, 1 - .95}
			break
		}
	}

	// Find a nice starting point for the plot.
	xMin := time.Second
	for xMin > 1 {
		if mmu := mmuCurve.MMU(xMin); mmu < 0.0001 {
			break
		}
		xMin /= 1000
	}
	// Cover six orders of magnitude.
	xMax := xMin * 1e6
	// But no more than the length of the trace.
	minEvent, maxEvent := mu[0][0].Time, mu[0][len(mu[0])-1].Time
	for _, mu1 := range mu[1:] {
		if mu1[0].Time < minEvent {
			minEvent = mu1[0].Time
		}
		if mu1[len(mu1)-1].Time > maxEvent {
			maxEvent = mu1[len(mu1)-1].Time
		}
	}
	if maxMax := time.Duration(maxEvent - minEvent); xMax > maxMax {
		xMax = maxMax
	}
	// Compute MMU curve.
	logMin, logMax := math.Log(float64(xMin)), math.Log(float64(xMax))
	const samples = 100
	plot := make([][]float64, samples)
	for i := 0; i < samples; i++ {
		window := time.Duration(math.Exp(float64(i)/(samples-1)*(logMax-logMin) + logMin))
		if quantiles == nil {
			plot[i] = make([]float64, 2)
			plot[i][1] = mmuCurve.MMU(window)
		} else {
			plot[i] = make([]float64, 1+len(quantiles))
			copy(plot[i][1:], mmuCurve.MUD(window, quantiles))
		}
		plot[i][0] = float64(window)
	}

	// Create JSON response.
	err = json.NewEncoder(w).Encode(map[string]interface{}{"xMin": int64(xMin), "xMax": int64(xMax), "quantiles": quantiles, "curve": plot})
	if err != nil {
		log.Printf("failed to serialize response: %v", err)
		return
	}
}

var templMMU = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      var chartsReady = false;
      google.charts.setOnLoadCallback(function() { chartsReady = true; refreshChart(); });

      var chart;
      var curve;

      function niceDuration(ns) {
          if (ns < 1e3) { return ns + 'ns'; }
          else if (ns < 1e6) { return ns / 1e3 + 'µs'; }
          else if (ns < 1e9) { return ns / 1e6 + 'ms'; }
          else { return ns / 1e9 + 's'; }
      }

      function niceQuantile(q) {
        return 'p' + q*100;
      }

      function mmuFlags() {
        var flags = "";
        $("#options input").each(function(i, elt) {
          if (elt.checked)
            flags += "|" + elt.id;
        });
        return flags.substr(1);
      }

      function refreshChart() {
        if (!chartsReady) return;
        var container = $('#mmu_chart');
        container.css('opacity', '.5');
        refreshChart.count++;
        var seq = refreshChart.count;
        $.getJSON('/mmuPlot?flags=' + mmuFlags())
         .fail(function(xhr, status, error) {
           alert('failed to load plot: ' + status);
         })
         .done(function(result) {
           if (refreshChart.count === seq)
             drawChart(result);
         });
      }
      refreshChart.count = 0;

      function drawChart(plotData) {
        curve = plotData.curve;
        var data = new google.visualization.DataTable();
        data.addColumn('number', 'Window duration');
        data.addColumn('number', 'Minimum mutator utilization');
        if (plotData.quantiles) {
          for (var i = 1; i < plotData.quantiles.length; i++) {
            data.addColumn('number', niceQuantile(1 - plotData.quantiles[i]) + ' MU');
          }
        }
        data.addRows(curve);
        for (var i = 0; i < curve.length; i++) {
          data.setFormattedValue(i, 0, niceDuration(curve[i][0]));
        }

        var options = {
          chart: {
            title: 'Minimum mutator utilization',
          },
          hAxis: {
            title: 'Window duration',
            scaleType: 'log',
            ticks: [],
          },
          vAxis: {
            title: 'Minimum mutator utilization',
            minValue: 0.0,
            maxValue: 1.0,
          },
          legend: { position: 'none' },
          focusTarget: 'category',
          width: 900,
          height: 500,
          chartArea: { width: '80%', height: '80%' },
        };
        for (var v = plotData.xMin; v <= plotData.xMax; v *= 10) {
          options.hAxis.ticks.push({v:v, f:niceDuration(v)});
        }
        if (plotData.quantiles) {
          options.vAxis.title = 'Mutator utilization';
          options.legend.position = 'in';
        }

        var container = $('#mmu_chart');
        container.empty();
        container.css('opacity', '');
        chart = new google.visualization.LineChart(container[0]);
        chart = new google.visualization.LineChart(document.getElementById('mmu_chart'));
        chart.draw(data, options);

        google.visualization.events.addListener(chart, 'select', selectHandler);
        $('#details').empty();
      }

      function selectHandler() {
        var items = chart.getSelection();
        if (items.length === 0) {
          return;
        }
        var details = $('#details');
        details.empty();
        var windowNS = curve[items[0].row][0];
        var url = '/mmuDetails?window=' + windowNS + '&flags=' + mmuFlags();
        $.getJSON(url)
         .fail(function(xhr, status, error) {
            details.text(status + ': ' + url + ' could not be loaded');
         })
         .done(function(worst) {
            details.text('Lowest mutator utilization in ' + niceDuration(windowNS) + ' windows:');
            for (var i = 0; i < worst.length; i++) {
              details.append($('<br/>'));
              var text = worst[i].MutatorUtil.toFixed(3) + ' at time ' + niceDuration(worst[i].Time);
              details.append($('<a/>').text(text).attr('href', worst[i].URL));
            }
         });
      }

      $.when($.ready).then(function() {
        $("#options input").click(refreshChart);
      });
    </script>
    <style>
      .help {
        display: inline-block;
        position: relative;
        width: 1em;
        height: 1em;
        border-radius: 50%;
        color: #fff;
        background: #555;
        text-align: center;
        cursor: help;
      }
      .help > span {
        display: none;
      }
      .help:hover > span {
        display: block;
        position: absolute;
        left: 1.1em;
        top: 1.1em;
        background: #555;
        text-align: left;
        width: 20em;
        padding: 0.5em;
        border-radius: 0.5em;
        z-index: 5;
      }
    </style>
  </head>
  <body>
    <div style="position: relative">
      <div id="mmu_chart" style="width: 900px; height: 500px; display: inline-block; vertical-align: top">Loading plot...</div>
      <div id="options" style="display: inline-block; vertical-align: top">
        <p>
          <b>View</b><br/>
          <input type="radio" name="view" id="system" checked><label for="system">System</label>
          <span class="help">?<span>Consider whole system utilization. For example, if one of four procs is available to the mutator, mutator utilization will be 0.25. This is the standard definition of an MMU.</span></span><br/>
          <input type="radio" name="view" id="perProc"><label for="perProc">Per-goroutine</label>
          <span class="help">?<span>Consider per-goroutine utilization. When even one goroutine is interrupted by GC, mutator utilization is 0.</span></span><br/>
        </p>
        <p>
          <b>Include</b><br/>
          <input type="checkbox" id="stw" checked><label for="stw">STW</label>
          <span class="help">?<span>Stop-the-world stops all goroutines simultaneously.</span></span><br/>
          <input type="checkbox" id="background" checked><label for="background">Background workers</label>
          <span class="help">?<span>Background workers are GC-specific goroutines. 25% of the CPU is dedicated to background workers during GC.</span></span><br/>
          <input type="checkbox" id="assist" checked><label for="assist">Mark assist</label>
          <span class="help">?<span>Mark assists are performed by allocation to prevent the mutator from outpacing GC.</span></span><br/>
          <input type="checkbox" id="sweep"><label for="sweep">Sweep</label>
          <span class="help">?<span>Sweep reclaims unused memory between GCs. (Enabling this may be very slow.).</span></span><br/>
        </p>
        <p>
          <b>Display</b><br/>
          <input type="checkbox" id="mut"><label for="mut">Show percentiles</label>
          <span class="help">?<span>Display percentile mutator utilization in addition to minimum. E.g., p99 MU drops the worst 1% of windows.</span></span><br/>
        </p>
      </div>
    </div>
    <div id="details">Select a point for details.</div>
  </body>
</html>
`

// httpMMUDetails serves details of an MMU graph at a particular window.
func httpMMUDetails(w http.ResponseWriter, r *http.Request) {
	_, mmuCurve, err := getMMUCurve(r)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse events: %v", err), http.StatusInternalServerError)
		return
	}

	windowStr := r.FormValue("window")
	window, err := strconv.ParseUint(windowStr, 10, 64)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse window parameter %q: %v", windowStr, err), http.StatusBadRequest)
		return
	}
	worst := mmuCurve.Examples(time.Duration(window), 10)

	// Construct a link for each window.
	var links []linkedUtilWindow
	for _, ui := range worst {
		links = append(links, newLinkedUtilWindow(ui, time.Duration(window)))
	}

	err = json.NewEncoder(w).Encode(links)
	if err != nil {
		log.Printf("failed to serialize trace: %v", err)
		return
	}
}

type linkedUtilWindow struct {
	trace.UtilWindow
	URL string
}

func newLinkedUtilWindow(ui trace.UtilWindow, window time.Duration) linkedUtilWindow {
	// Find the range containing this window.
	var r Range
	for _, r = range ranges {
		if r.EndTime > ui.Time {
			break
		}
	}
	return linkedUtilWindow{ui, fmt.Sprintf("%s#%v:%v", r.URL(), float64(ui.Time)/1e6, float64(ui.Time+int64(window))/1e6)}
}
