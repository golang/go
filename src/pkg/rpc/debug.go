// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

/*
	Some HTML presented at http://machine:port/debug/rpc
	Lists services, their methods, and some statistics, still rudimentary.
*/

import (
	"fmt";
	"http";
	"os";
	"sort";
	"template";
)

const debugText = `<html>
	<body>
	<title>Services</title>
	{.repeated section @}
	<hr>
	Service {name}
	<hr>
		<table>
		<th align=center>Method</th><th align=center>Calls</th>
		{.repeated section meth}
			<tr>
			<td align=left font=fixed>{name}({m.argType}, {m.replyType}) os.Error</td>
			<td align=center>{m.numCalls}</td>
			</tr>
		{.end}
		</table>
	{.end}
	</body>
	</html>`

var debug *template.Template

type debugMethod struct {
	m	*methodType;
	name	string;
}

type methodArray []debugMethod

type debugService struct {
	s	*service;
	name	string;
	meth	methodArray;
}

type serviceArray []debugService

func (s serviceArray) Len() int {
	return len(s);
}
func (s serviceArray) Less(i, j int) bool {
	return s[i].name < s[j].name;
}
func (s serviceArray) Swap(i, j int) {
	s[i], s[j] = s[j], s[i];
}

func (m methodArray) Len() int {
	return len(m);
}
func (m methodArray) Less(i, j int) bool {
	return m[i].name < m[j].name;
}
func (m methodArray) Swap(i, j int) {
	m[i], m[j] = m[j], m[i];
}

// Runs at /debug/rpc
func debugHTTP(c *http.Conn, req *http.Request) {
	var err os.Error;
	if debug == nil {
		debug, err = template.Parse(debugText, nil);
		if err != nil {
			fmt.Fprintln(c, "rpc can't create debug HTML template:", err.String());
			return;
		}
	}
	// Build a sorted version of the data.
	var services = make(serviceArray, len(server.serviceMap));
	i := 0;
	server.Lock();
	for sname, service := range server.serviceMap {
		services[i] = debugService{service, sname, make(methodArray, len(service.method))};
		j := 0;
		for mname, method := range service.method {
			services[i].meth[j] = debugMethod{method, mname};
			j++;
		}
		sort.Sort(services[i].meth);
		i++;
	}
	server.Unlock();
	sort.Sort(services);
	err = debug.Execute(services, c);
	if err != nil {
		fmt.Fprintln(c, "rpc: error executing template:", err.String());
	}
}
