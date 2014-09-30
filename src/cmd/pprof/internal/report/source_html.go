// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package report

const weblistPageHeader = `
<!DOCTYPE html>
<html>
<head>
<title>Pprof listing</title>
<style type="text/css">
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
.line {
color: #aaaaaa;
}
.nop {
color: #aaaaaa;
}
.unimportant {
color: #cccccc;
}
.disasmloc {
color: #000000;
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
</style>
<script type="text/javascript">
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
</script>
</head>
<body>
`

const weblistPageClosing = `
</body>
</html>
`
