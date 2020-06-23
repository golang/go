// Copyright 2014 Google Inc. All Rights Reserved.
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

package report

import (
	"html/template"
)

// AddSourceTemplates adds templates used by PrintWebList to t.
func AddSourceTemplates(t *template.Template) {
	template.Must(t.Parse(`{{define "weblistcss"}}` + weblistPageCSS + `{{end}}`))
	template.Must(t.Parse(`{{define "weblistjs"}}` + weblistPageScript + `{{end}}`))
}

const weblistPageCSS = `<style type="text/css">
body {
font-family: sans-serif;
}
h1 {
  font-size: 1.5em;
  margin-bottom: 4px;
}
.legend {
  font-size: 1.25em;
}
.line, .nop, .unimportant {
  color: #aaaaaa;
}
.inlinesrc {
  color: #000066;
}
.deadsrc {
cursor: pointer;
}
.deadsrc:hover {
background-color: #eeeeee;
}
.livesrc {
color: #0000ff;
cursor: pointer;
}
.livesrc:hover {
background-color: #eeeeee;
}
.asm {
color: #008800;
display: none;
}
</style>`

const weblistPageScript = `<script type="text/javascript">
function pprof_toggle_asm(e) {
  var target;
  if (!e) e = window.event;
  if (e.target) target = e.target;
  else if (e.srcElement) target = e.srcElement;

  if (target) {
    var asm = target.nextSibling;
    if (asm && asm.className == "asm") {
      asm.style.display = (asm.style.display == "block" ? "" : "block");
      e.preventDefault();
      return false;
    }
  }
}
</script>`

const weblistPageClosing = `
</body>
</html>
`
