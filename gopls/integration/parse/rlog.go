// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"fmt"
	"log"
	"strings"
)

// Rlog contains the processed logs
type Rlog struct {
	Logs         []*Logmsg          // In the order in the log file
	ServerCall   map[string]*Logmsg // ID->Request, client->server
	ServerReply  map[string]*Logmsg // ID->Response, server->client (includes Errors)
	ClientCall   map[string]*Logmsg
	ClientReply  map[string]*Logmsg
	ClientNotifs []*Logmsg
	ServerNotifs []*Logmsg
	Histogram    *LogHist
}

func newRlog(x []*Logmsg) *Rlog {
	return &Rlog{Logs: x,
		ServerCall:   make(map[string]*Logmsg),
		ServerReply:  make(map[string]*Logmsg),
		ClientCall:   make(map[string]*Logmsg),
		ClientReply:  make(map[string]*Logmsg),
		ClientNotifs: []*Logmsg{},
		ServerNotifs: []*Logmsg{},
		Histogram:    &LogHist{},
	}
}

// Counts returns a one-line summary of an Rlog
func (r *Rlog) Counts() string {
	return fmt.Sprintf("logs:%d srvC:%d srvR:%d clC:%d clR:%d clN:%d srvN:%d",
		len(r.Logs),
		len(r.ServerCall), len(r.ServerReply), len(r.ClientCall), len(r.ClientReply),
		len(r.ClientNotifs), len(r.ServerNotifs))
}

// ToRlog reads a log file and returns a *Rlog
func ToRlog(fname string) (*Rlog, error) {
	x, err := ReadLogs(fname)
	if err != nil {
		return nil, err
	}
	ans := newRlog(x)
	for _, l := range x {
		switch l.Type {
		case ClRequest:
			ans.ServerCall[l.ID] = l
		case ClResponse:
			ans.ServerReply[l.ID] = l
			if l.Type != ReportErr {
				n := 0
				fmt.Sscanf(l.Elapsed, "%d", &n)
				ans.Histogram.add(n)
			}
		case SvRequest:
			ans.ClientCall[l.ID] = l
		case SvResponse:
			ans.ClientReply[l.ID] = l
		case ToClient:
			ans.ClientNotifs = append(ans.ClientNotifs, l)
		case ToServer:
			ans.ServerNotifs = append(ans.ServerNotifs, l)
		case ReportErr:
			ans.ServerReply[l.ID] = l
			l.Method = ans.ServerCall[l.ID].Method // Method not in log message
		default:
			log.Fatalf("eh? %s/%s (%s)", l.Type, l.Method, l.ID)
		}
	}
	return ans, nil
}

// LogHist gets ints, and puts them into buckets:
// <=10, <=30, 100, 300, 1000, ...
// It produces a historgram of elapsed times in milliseconds
type LogHist struct {
	cnts []int
}

func (l *LogHist) add(n int) {
	if n < 0 {
		n = 0
	}
	bucket := 0
	for ; n > 0; n /= 10 {
		if n < 10 {
			break
		}
		if n < 30 {
			bucket++
			break
		}
		bucket += 2
	}
	if len(l.cnts) <= bucket {
		for j := len(l.cnts); j < bucket+10; j++ {
			l.cnts = append(l.cnts, 0)
		}
	}
	l.cnts[bucket]++
}

// String returns a string describing a histogram
func (l *LogHist) String() string {
	top := len(l.cnts) - 1
	for ; top > 0 && l.cnts[top] == 0; top-- {
	}
	labs := []string{"10", "30"}
	out := strings.Builder{}
	out.WriteByte('[')
	for i := 0; i <= top; i++ {
		label := labs[i%2]
		labs[i%2] += "0"
		fmt.Fprintf(&out, "%s:%d ", label, l.cnts[i])
	}
	out.WriteByte(']')
	return out.String()
}
