// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// This program converts CSV calibration data printed by
//
//	go test -run=Calibrate/Name -calibrate >file.csv
//
// into an SVG file. Invoke as:
//
//	go run calibrate_graph.go file.csv >file.svg
//
// See calibrate.md for more details.

package main

import (
	"bytes"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run calibrate_graph.go file.csv >file.svg\n")
	os.Exit(2)
}

// A Point is an X, Y coordinate in the data being plotted.
type Point struct {
	X, Y float64
}

// A Graph is a graph to draw as SVG.
type Graph struct {
	Title   string    // title above graph
	Geomean []Point   // geomean line
	Lines   [][]Point // normalized data lines
	XAxis   string    // x-axis label
	YAxis   string    // y-axis label
	Min     Point     // min point of data display
	Max     Point     // max point of data display
}

var yMax = flag.Float64("ymax", 1.2, "maximum y axis value")
var alphaNorm = flag.Float64("alphanorm", 0.1, "alpha for a single norm line")

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}

	// Read CSV. It may be enclosed in
	//	-- name.csv --
	//	...
	//	-- eof --
	// framing, in which case remove the framing.
	fdata, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	if _, after, ok := bytes.Cut(fdata, []byte(".csv --\n")); ok {
		fdata = after
	}
	if before, _, ok := bytes.Cut(fdata, []byte("-- eof --\n")); ok {
		fdata = before
	}
	rd := csv.NewReader(bytes.NewReader(fdata))
	rd.FieldsPerRecord = -1
	records, err := rd.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Construct graph from loaded CSV.
	// CSV starts with metadata lines like
	//	goos,darwin
	// and then has two tables of timings.
	// Each table looks like
	//	size \ threshold,10,20,30,40
	//	100,1,2,3,4
	//	200,2,3,4,5
	//	300,3,4,5,6
	//	400,4,5,6,7
	//	500,5,6,7,8
	// The header line gives the threshold values and then each row
	// gives an input size and the timings for each threshold.
	// Omitted timings are empty strings and turn into infinities when parsing.
	// The first table gives raw nanosecond timings.
	// The second table gives timings normalized relative to the fastest
	// possible threshold for a given input size.
	// We only want the second table.
	// The tables are followed by a list of geomeans of all the normalized
	// timings for each threshold:
	//	geomean,1.2,1.1,1.0,1.4
	// We turn each normalized timing row into a line in the graph,
	// and we turn the geomean into an overlaid thick line.
	// The metadata is used for preparing the titles.
	g := &Graph{
		YAxis: "Relative Slowdown",
		Min:   Point{0, 1},
		Max:   Point{1, 1.2},
	}
	meta := make(map[string]string)
	table := 0 // number of table headers seen
	var thresholds []float64
	maxNorm := 0.0
	for _, rec := range records {
		if len(rec) == 0 {
			continue
		}
		if len(rec) == 2 {
			meta[rec[0]] = rec[1]
			continue
		}
		if rec[0] == `size \ threshold` {
			table++
			if table == 2 {
				thresholds = parseFloats(rec)
				g.Min.X = thresholds[0]
				g.Max.X = thresholds[len(thresholds)-1]
			}
			continue
		}
		if rec[0] == "geomean" {
			table = 3 // end of norms table
			geomeans := parseFloats(rec)
			g.Geomean = floatsToLine(thresholds, geomeans)
			continue
		}
		if table == 2 {
			if _, err := strconv.Atoi(rec[0]); err != nil { // size
				log.Fatalf("invalid table line: %q", rec)
			}
			norms := parseFloats(rec)
			if len(norms) > len(thresholds) {
				log.Fatalf("too many timings (%d > %d): %q", len(norms), len(thresholds), rec)
			}
			g.Lines = append(g.Lines, floatsToLine(thresholds, norms))
			for _, y := range norms {
				maxNorm = max(maxNorm, y)
			}
			continue
		}
	}

	g.Max.Y = min(*yMax, math.Ceil(maxNorm*100)/100)
	g.XAxis = meta["calibrate"] + "Threshold"
	g.Title = meta["goos"] + "/" + meta["goarch"] + " " + meta["cpu"]

	os.Stdout.Write(g.SVG())
}

// parseFloats parses rec[1:] as floating point values.
// If a field is the empty string, it is represented as +Inf.
func parseFloats(rec []string) []float64 {
	floats := make([]float64, 0, len(rec)-1)
	for _, v := range rec[1:] {
		if v == "" {
			floats = append(floats, math.Inf(+1))
			continue
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			log.Fatalf("invalid record: %q (%v)", rec, err)
		}
		floats = append(floats, f)
	}
	return floats
}

// floatsToLine converts a sequence of floats into a line, ignoring missing (infinite) values.
func floatsToLine(x, y []float64) []Point {
	var line []Point
	for i, yi := range y {
		if !math.IsInf(yi, 0) {
			line = append(line, Point{x[i], yi})
		}
	}
	return line
}

const svgHeader = `<svg width="%d" height="%d" version="1.1" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style type="text/css"><![CDATA[
      text { stroke-width: 0; white-space: pre; }
      text.hjc { text-anchor: middle; }
      text.hjl { text-anchor: start; }
      text.hjr { text-anchor: end; }
      .def { stroke-linecap: round; stroke-linejoin: round; fill: none; stroke: #000000; stroke-width: 1px; }
      .tick { stroke: #000000; fill: #000000; font: %dpx Times; }
      .title { stroke: #000000; fill: #000000; font: %dpx Times; font-weight: bold; }
      .axis { stroke-width: 2px; }
      .norm { stroke: rgba(0,0,0,%f); }
      .geomean { stroke: #6666ff; stroke-width: 2px; }
    ]]></style>
  </defs>
  <g class="def">
`

// Layout constants for drawing graph
const (
	DX   = 600          // width of graphed data
	DY   = 150          // height of graphed data
	ML   = 80           // margin left
	MT   = 30           // margin top
	MR   = 10           // margin right
	MB   = 50           // margin bottom
	PS   = 14           // point size of text
	W    = ML + DX + MR // width of overall graph
	H    = MT + DY + MB // height of overall graph
	Tick = 5            // axis tick length
)

// An SVGPoint is a point in the SVG image, in pixel units,
// with Y increasing down the page.
type SVGPoint struct {
	X, Y int
}

func (p SVGPoint) String() string {
	return fmt.Sprintf("%d,%d", p.X, p.Y)
}

// pt converts an x, y data value (such as from a Point) to an SVGPoint.
func (g *Graph) pt(x, y float64) SVGPoint {
	return SVGPoint{
		X: ML + int((x-g.Min.X)/(g.Max.X-g.Min.X)*DX),
		Y: H - MB - int((y-g.Min.Y)/(g.Max.Y-g.Min.Y)*DY),
	}
}

// SVG returns the SVG text for the graph.
func (g *Graph) SVG() []byte {

	var svg bytes.Buffer
	fmt.Fprintf(&svg, svgHeader, W, H, PS, PS, *alphaNorm)

	// Draw data, clipped.
	fmt.Fprintf(&svg, "<clipPath id=\"cp\"><path d=\"M %v L %v L %v L %v Z\" /></clipPath>\n",
		g.pt(g.Min.X, g.Min.Y), g.pt(g.Max.X, g.Min.Y), g.pt(g.Max.X, g.Max.Y), g.pt(g.Min.X, g.Max.Y))
	fmt.Fprintf(&svg, "<g clip-path=\"url(#cp)\">\n")
	for _, line := range g.Lines {
		if len(line) == 0 {
			continue
		}
		fmt.Fprintf(&svg, "<path class=\"norm\" d=\"M %v", g.pt(line[0].X, line[0].Y))
		for _, v := range line[1:] {
			fmt.Fprintf(&svg, " L %v", g.pt(v.X, v.Y))
		}
		fmt.Fprintf(&svg, "\"/>\n")
	}
	// Draw geomean.
	if len(g.Geomean) > 0 {
		line := g.Geomean
		fmt.Fprintf(&svg, "<path class=\"geomean\" d=\"M %v", g.pt(line[0].X, line[0].Y))
		for _, v := range line[1:] {
			fmt.Fprintf(&svg, " L %v", g.pt(v.X, v.Y))
		}
		fmt.Fprintf(&svg, "\"/>\n")
	}
	fmt.Fprintf(&svg, "</g>\n")

	// Draw axes and major and minor tick marks.
	fmt.Fprintf(&svg, "<path class=\"axis\" d=\"")
	fmt.Fprintf(&svg, " M %v L %v", g.pt(g.Min.X, g.Min.Y), g.pt(g.Max.X, g.Min.Y)) // x axis
	fmt.Fprintf(&svg, " M %v L %v", g.pt(g.Min.X, g.Min.Y), g.pt(g.Min.X, g.Max.Y)) // y axis
	xscale := 10.0
	if g.Max.X-g.Min.X < 100 {
		xscale = 1.0
	}
	for x := int(math.Ceil(g.Min.X / xscale)); float64(x)*xscale <= g.Max.X; x++ {
		if x%5 != 0 {
			fmt.Fprintf(&svg, " M %v l 0,%d", g.pt(float64(x)*xscale, g.Min.Y), Tick)
		} else {
			fmt.Fprintf(&svg, " M %v l 0,%d", g.pt(float64(x)*xscale, g.Min.Y), 2*Tick)
		}
	}
	yscale := 100.0
	if g.Max.Y-g.Min.Y > 0.5 {
		yscale = 10
	}
	for y := int(math.Ceil(g.Min.Y * yscale)); float64(y) <= g.Max.Y*yscale; y++ {
		if y%5 != 0 {
			fmt.Fprintf(&svg, " M %v l -%d,0", g.pt(g.Min.X, float64(y)/yscale), Tick)
		} else {
			fmt.Fprintf(&svg, " M %v l -%d,0", g.pt(g.Min.X, float64(y)/yscale), 2*Tick)
		}
	}
	fmt.Fprintf(&svg, "\"/>\n")

	// Draw tick labels on major marks.
	for x := int(math.Ceil(g.Min.X / xscale)); float64(x)*xscale <= g.Max.X; x++ {
		if x%5 == 0 {
			p := g.pt(float64(x)*xscale, g.Min.Y)
			fmt.Fprintf(&svg, "<text x=\"%d\" y=\"%d\" class=\"tick hjc\">%d</text>\n", p.X, p.Y+2*Tick+PS, x*int(xscale))
		}
	}
	for y := int(math.Ceil(g.Min.Y * yscale)); float64(y) <= g.Max.Y*yscale; y++ {
		if y%5 == 0 {
			p := g.pt(g.Min.X, float64(y)/yscale)
			fmt.Fprintf(&svg, "<text x=\"%d\" y=\"%d\" class=\"tick hjr\">%.2f</text>\n", p.X-2*Tick-Tick, p.Y+PS/3, float64(y)/yscale)
		}
	}

	// Draw graph title and axis titles.
	fmt.Fprintf(&svg, "<text x=\"%d\" y=\"%d\" class=\"title hjc\">%s</text>\n", ML+DX/2, MT-PS/3, g.Title)
	fmt.Fprintf(&svg, "<text x=\"%d\" y=\"%d\" class=\"title hjc\">%s</text>\n", ML+DX/2, MT+DY+2*Tick+2*PS+PS/2, g.XAxis)
	fmt.Fprintf(&svg, "<g transform=\"translate(%d,%d) rotate(-90)\"><text x=\"0\" y=\"0\" class=\"title hjc\">%s</text></g>\n", ML-Tick-Tick-3*PS, MT+DY/2, g.YAxis)

	fmt.Fprintf(&svg, "</g></svg>\n")
	return svg.Bytes()
}
