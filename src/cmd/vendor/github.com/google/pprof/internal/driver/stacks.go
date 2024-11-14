// Copyright 2022 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"encoding/json"
	"html/template"
	"net/http"

	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/internal/report"
)

// stackView generates the flamegraph view.
func (ui *webInterface) stackView(w http.ResponseWriter, req *http.Request) {
	// Get all data in a report.
	rpt, errList := ui.makeReport(w, req, []string{"svg"}, func { cfg ->
		cfg.CallTree = true
		cfg.Trim = false
		cfg.Granularity = "filefunctions"
	})
	if rpt == nil {
		return // error already reported
	}

	// Make stack data and generate corresponding JSON.
	stacks := rpt.Stacks()
	b, err := json.Marshal(stacks)
	if err != nil {
		http.Error(w, "error serializing stacks for flame graph",
			http.StatusInternalServerError)
		ui.options.UI.PrintErr(err)
		return
	}

	nodes := make([]string, len(stacks.Sources))
	for i, src := range stacks.Sources {
		nodes[i] = src.FullName
	}
	nodes[0] = "" // root is not a real node

	_, legend := report.TextItems(rpt)
	ui.render(w, req, "stacks", rpt, errList, legend, webArgs{
		Stacks:   template.JS(b),
		Nodes:    nodes,
		UnitDefs: measurement.UnitTypes,
	})
}
