// Copyright 2017 Google Inc. All Rights Reserved.
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

import "html/template"

import "github.com/google/pprof/third_party/d3"
import "github.com/google/pprof/third_party/d3flamegraph"

// addTemplates adds a set of template definitions to templates.
func addTemplates(templates *template.Template) {
	template.Must(templates.Parse(`{{define "d3script"}}` + d3.JSSource + `{{end}}`))
	template.Must(templates.Parse(`{{define "d3flamegraphscript"}}` + d3flamegraph.JSSource + `{{end}}`))
	template.Must(templates.Parse(`{{define "d3flamegraphcss"}}` + d3flamegraph.CSSSource + `{{end}}`))
	template.Must(templates.Parse(`
{{define "css"}}
<style type="text/css">
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
html, body {
  height: 100%;
}
body {
  font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
  font-size: 13px;
  line-height: 1.4;
  display: flex;
  flex-direction: column;
}
a {
  color: #2a66d9;
}
.header {
  display: flex;
  align-items: center;
  height: 44px;
  min-height: 44px;
  background-color: #eee;
  color: #212121;
  padding: 0 1rem;
}
.header > div {
  margin: 0 0.125em;
}
.header .title h1 {
  font-size: 1.75em;
  margin-right: 1rem;
}
.header .title a {
  color: #212121;
  text-decoration: none;
}
.header .title a:hover {
  text-decoration: underline;
}
.header .description {
  width: 100%;
  text-align: right;
  white-space: nowrap;
}
@media screen and (max-width: 799px) {
  .header input {
    display: none;
  }
}
#detailsbox {
  display: none;
  z-index: 1;
  position: fixed;
  top: 40px;
  right: 20px;
  background-color: #ffffff;
  box-shadow: 0 1px 5px rgba(0,0,0,.3);
  line-height: 24px;
  padding: 1em;
  text-align: left;
}
.header input {
  background: white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' style='pointer-events:none;display:block;width:100%25;height:100%25;fill:#757575'%3E%3Cpath d='M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61.0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z'/%3E%3C/svg%3E") no-repeat 4px center/20px 20px;
  border: 1px solid #d1d2d3;
  border-radius: 2px 0 0 2px;
  padding: 0.25em;
  padding-left: 28px;
  margin-left: 1em;
  font-family: 'Roboto', 'Noto', sans-serif;
  font-size: 1em;
  line-height: 24px;
  color: #212121;
}
.downArrow {
  border-top: .36em solid #ccc;
  border-left: .36em solid transparent;
  border-right: .36em solid transparent;
  margin-bottom: .05em;
  margin-left: .5em;
  transition: border-top-color 200ms;
}
.menu-item {
  height: 100%;
  text-transform: uppercase;
  font-family: 'Roboto Medium', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
  position: relative;
}
.menu-item .menu-name:hover {
  opacity: 0.75;
}
.menu-item .menu-name:hover .downArrow {
  border-top-color: #666;
}
.menu-name {
  height: 100%;
  padding: 0 0.5em;
  display: flex;
  align-items: center;
  justify-content: center;
}
.submenu {
  display: none;
  z-index: 1;
  margin-top: -4px;
  min-width: 10em;
  position: absolute;
  left: 0px;
  background-color: white;
  box-shadow: 0 1px 5px rgba(0,0,0,.3);
  font-size: 100%;
  text-transform: none;
}
.menu-item, .submenu {
  user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  -webkit-user-select: none;
}
.submenu hr {
  border: 0;
  border-top: 2px solid #eee;
}
.submenu a {
  display: block;
  padding: .5em 1em;
  text-decoration: none;
}
.submenu a:hover, .submenu a.active {
  color: white;
  background-color: #6b82d6;
}
.submenu a.disabled {
  color: gray;
  pointer-events: none;
}

#content {
  overflow-y: scroll;
  padding: 1em;
}
#top {
  overflow-y: scroll;
}
#graph {
  overflow: hidden;
}
#graph svg {
  width: 100%;
  height: auto;
  padding: 10px;
}
#content.source .filename {
  margin-top: 0;
  margin-bottom: 1em;
  font-size: 120%;
}
#content.source pre {
  margin-bottom: 3em;
}
table {
  border-spacing: 0px;
  width: 100%;
  padding-bottom: 1em;
  white-space: nowrap;
}
table thead {
  font-family: 'Roboto Medium', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
}
table tr th {
  background-color: #ddd;
  text-align: right;
  padding: .3em .5em;
}
table tr td {
  padding: .3em .5em;
  text-align: right;
}
#top table tr th:nth-child(6),
#top table tr th:nth-child(7),
#top table tr td:nth-child(6),
#top table tr td:nth-child(7) {
  text-align: left;
}
#top table tr td:nth-child(6) {
  width: 100%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}
#flathdr1, #flathdr2, #cumhdr1, #cumhdr2, #namehdr {
  cursor: ns-resize;
}
.hilite {
  background-color: #ebf5fb;
  font-weight: bold;
}
</style>
{{end}}

{{define "header"}}
<div class="header">
  <div class="title">
    <h1><a href="./">pprof</a></h1>
  </div>

  <div id="view" class="menu-item">
    <div class="menu-name">
      View
      <i class="downArrow"></i>
    </div>
    <div class="submenu">
      <a title="{{.Help.top}}"  href="./top" id="topbtn">Top</a>
      <a title="{{.Help.graph}}" href="./" id="graphbtn">Graph</a>
      <a title="{{.Help.flamegraph}}" href="./flamegraph" id="flamegraph">Flame Graph</a>
      <a title="{{.Help.peek}}" href="./peek" id="peek">Peek</a>
      <a title="{{.Help.list}}" href="./source" id="list">Source</a>
      <a title="{{.Help.disasm}}" href="./disasm" id="disasm">Disassemble</a>
    </div>
  </div>

  {{$sampleLen := len .SampleTypes}}
  {{if gt $sampleLen 1}}
  <div id="sample" class="menu-item">
    <div class="menu-name">
      Sample
      <i class="downArrow"></i>
    </div>
    <div class="submenu">
      {{range .SampleTypes}}
      <a href="?si={{.}}" id="{{.}}">{{.}}</a>
      {{end}}
    </div>
  </div>
  {{end}}

  <div id="refine" class="menu-item">
    <div class="menu-name">
      Refine
      <i class="downArrow"></i>
    </div>
    <div class="submenu">
      <a title="{{.Help.focus}}" href="?" id="focus">Focus</a>
      <a title="{{.Help.ignore}}" href="?" id="ignore">Ignore</a>
      <a title="{{.Help.hide}}" href="?" id="hide">Hide</a>
      <a title="{{.Help.show}}" href="?" id="show">Show</a>
      <a title="{{.Help.show_from}}" href="?" id="show-from">Show from</a>
      <hr>
      <a title="{{.Help.reset}}" href="?">Reset</a>
    </div>
  </div>

  <div>
    <input id="search" type="text" placeholder="Search regexp" autocomplete="off" autocapitalize="none" size=40>
  </div>

  <div class="description">
    <a title="{{.Help.details}}" href="#" id="details">{{.Title}}</a>
    <div id="detailsbox">
      {{range .Legend}}<div>{{.}}</div>{{end}}
    </div>
  </div>
</div>

<div id="errors">{{range .Errors}}<div>{{.}}</div>{{end}}</div>
{{end}}

{{define "graph" -}}
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{.Title}}</title>
  {{template "css" .}}
</head>
<body>
  {{template "header" .}}
  <div id="graph">
    {{.HTMLBody}}
  </div>
  {{template "script" .}}
  <script>viewer(new URL(window.location.href), {{.Nodes}});</script>
</body>
</html>
{{end}}

{{define "script"}}
<script>
// Make svg pannable and zoomable.
// Call clickHandler(t) if a click event is caught by the pan event handlers.
function initPanAndZoom(svg, clickHandler) {
  'use strict';

  // Current mouse/touch handling mode
  const IDLE = 0;
  const MOUSEPAN = 1;
  const TOUCHPAN = 2;
  const TOUCHZOOM = 3;
  let mode = IDLE;

  // State needed to implement zooming.
  let currentScale = 1.0;
  const initWidth = svg.viewBox.baseVal.width;
  const initHeight = svg.viewBox.baseVal.height;

  // State needed to implement panning.
  let panLastX = 0;      // Last event X coordinate
  let panLastY = 0;      // Last event Y coordinate
  let moved = false;     // Have we seen significant movement
  let touchid = null;    // Current touch identifier

  // State needed for pinch zooming
  let touchid2 = null;     // Second id for pinch zooming
  let initGap = 1.0;       // Starting gap between two touches
  let initScale = 1.0;     // currentScale when pinch zoom started
  let centerPoint = null;  // Center point for scaling

  // Convert event coordinates to svg coordinates.
  function toSvg(x, y) {
    const p = svg.createSVGPoint();
    p.x = x;
    p.y = y;
    let m = svg.getCTM();
    if (m == null) m = svg.getScreenCTM(); // Firefox workaround.
    return p.matrixTransform(m.inverse());
  }

  // Change the scaling for the svg to s, keeping the point denoted
  // by u (in svg coordinates]) fixed at the same screen location.
  function rescale(s, u) {
    // Limit to a good range.
    if (s < 0.2) s = 0.2;
    if (s > 10.0) s = 10.0;

    currentScale = s;

    // svg.viewBox defines the visible portion of the user coordinate
    // system.  So to magnify by s, divide the visible portion by s,
    // which will then be stretched to fit the viewport.
    const vb = svg.viewBox;
    const w1 = vb.baseVal.width;
    const w2 = initWidth / s;
    const h1 = vb.baseVal.height;
    const h2 = initHeight / s;
    vb.baseVal.width = w2;
    vb.baseVal.height = h2;

    // We also want to adjust vb.baseVal.x so that u.x remains at same
    // screen X coordinate.  In other words, want to change it from x1 to x2
    // so that:
    //     (u.x - x1) / w1 = (u.x - x2) / w2
    // Simplifying that, we get
    //     (u.x - x1) * (w2 / w1) = u.x - x2
    //     x2 = u.x - (u.x - x1) * (w2 / w1)
    vb.baseVal.x = u.x - (u.x - vb.baseVal.x) * (w2 / w1);
    vb.baseVal.y = u.y - (u.y - vb.baseVal.y) * (h2 / h1);
  }

  function handleWheel(e) {
    if (e.deltaY == 0) return;
    // Change scale factor by 1.1 or 1/1.1
    rescale(currentScale * (e.deltaY < 0 ? 1.1 : (1/1.1)),
            toSvg(e.offsetX, e.offsetY));
  }

  function setMode(m) {
    mode = m;
    touchid = null;
    touchid2 = null;
  }

  function panStart(x, y) {
    moved = false;
    panLastX = x;
    panLastY = y;
  }

  function panMove(x, y) {
    let dx = x - panLastX;
    let dy = y - panLastY;
    if (Math.abs(dx) <= 2 && Math.abs(dy) <= 2) return; // Ignore tiny moves

    moved = true;
    panLastX = x;
    panLastY = y;

    // Firefox workaround: get dimensions from parentNode.
    const swidth = svg.clientWidth || svg.parentNode.clientWidth;
    const sheight = svg.clientHeight || svg.parentNode.clientHeight;

    // Convert deltas from screen space to svg space.
    dx *= (svg.viewBox.baseVal.width / swidth);
    dy *= (svg.viewBox.baseVal.height / sheight);

    svg.viewBox.baseVal.x -= dx;
    svg.viewBox.baseVal.y -= dy;
  }

  function handleScanStart(e) {
    if (e.button != 0) return; // Do not catch right-clicks etc.
    setMode(MOUSEPAN);
    panStart(e.clientX, e.clientY);
    e.preventDefault();
    svg.addEventListener('mousemove', handleScanMove);
  }

  function handleScanMove(e) {
    if (e.buttons == 0) {
      // Missed an end event, perhaps because mouse moved outside window.
      setMode(IDLE);
      svg.removeEventListener('mousemove', handleScanMove);
      return;
    }
    if (mode == MOUSEPAN) panMove(e.clientX, e.clientY);
  }

  function handleScanEnd(e) {
    if (mode == MOUSEPAN) panMove(e.clientX, e.clientY);
    setMode(IDLE);
    svg.removeEventListener('mousemove', handleScanMove);
    if (!moved) clickHandler(e.target);
  }

  // Find touch object with specified identifier.
  function findTouch(tlist, id) {
    for (const t of tlist) {
      if (t.identifier == id) return t;
    }
    return null;
  }

  // Return distance between two touch points
  function touchGap(t1, t2) {
    const dx = t1.clientX - t2.clientX;
    const dy = t1.clientY - t2.clientY;
    return Math.hypot(dx, dy);
  }

  function handleTouchStart(e) {
    if (mode == IDLE && e.changedTouches.length == 1) {
      // Start touch based panning
      const t = e.changedTouches[0];
      setMode(TOUCHPAN);
      touchid = t.identifier;
      panStart(t.clientX, t.clientY);
      e.preventDefault();
    } else if (mode == TOUCHPAN && e.touches.length == 2) {
      // Start pinch zooming
      setMode(TOUCHZOOM);
      const t1 = e.touches[0];
      const t2 = e.touches[1];
      touchid = t1.identifier;
      touchid2 = t2.identifier;
      initScale = currentScale;
      initGap = touchGap(t1, t2);
      centerPoint = toSvg((t1.clientX + t2.clientX) / 2,
                          (t1.clientY + t2.clientY) / 2);
      e.preventDefault();
    }
  }

  function handleTouchMove(e) {
    if (mode == TOUCHPAN) {
      const t = findTouch(e.changedTouches, touchid);
      if (t == null) return;
      if (e.touches.length != 1) {
        setMode(IDLE);
        return;
      }
      panMove(t.clientX, t.clientY);
      e.preventDefault();
    } else if (mode == TOUCHZOOM) {
      // Get two touches; new gap; rescale to ratio.
      const t1 = findTouch(e.touches, touchid);
      const t2 = findTouch(e.touches, touchid2);
      if (t1 == null || t2 == null) return;
      const gap = touchGap(t1, t2);
      rescale(initScale * gap / initGap, centerPoint);
      e.preventDefault();
    }
  }

  function handleTouchEnd(e) {
    if (mode == TOUCHPAN) {
      const t = findTouch(e.changedTouches, touchid);
      if (t == null) return;
      panMove(t.clientX, t.clientY);
      setMode(IDLE);
      e.preventDefault();
      if (!moved) clickHandler(t.target);
    } else if (mode == TOUCHZOOM) {
      setMode(IDLE);
      e.preventDefault();
    }
  }

  svg.addEventListener('mousedown', handleScanStart);
  svg.addEventListener('mouseup', handleScanEnd);
  svg.addEventListener('touchstart', handleTouchStart);
  svg.addEventListener('touchmove', handleTouchMove);
  svg.addEventListener('touchend', handleTouchEnd);
  svg.addEventListener('wheel', handleWheel, true);
}

function initMenus() {
  'use strict';

  let activeMenu = null;
  let activeMenuHdr = null;

  function cancelActiveMenu() {
    if (activeMenu == null) return;
    activeMenu.style.display = 'none';
    activeMenu = null;
    activeMenuHdr = null;
  }

  // Set click handlers on every menu header.
  for (const menu of document.getElementsByClassName('submenu')) {
    const hdr = menu.parentElement;
    if (hdr == null) return;
    if (hdr.classList.contains('disabled')) return;
    function showMenu(e) {
      // menu is a child of hdr, so this event can fire for clicks
      // inside menu. Ignore such clicks.
      if (e.target.parentElement != hdr) return;
      activeMenu = menu;
      activeMenuHdr = hdr;
      menu.style.display = 'block';
    }
    hdr.addEventListener('mousedown', showMenu);
    hdr.addEventListener('touchstart', showMenu);
  }

  // If there is an active menu and a down event outside, retract the menu.
  for (const t of ['mousedown', 'touchstart']) {
    document.addEventListener(t, (e) => {
      // Note: to avoid unnecessary flicker, if the down event is inside
      // the active menu header, do not retract the menu.
      if (activeMenuHdr != e.target.closest('.menu-item')) {
        cancelActiveMenu();
      }
    }, { passive: true, capture: true });
  }

  // If there is an active menu and an up event inside, retract the menu.
  document.addEventListener('mouseup', (e) => {
    if (activeMenu == e.target.closest('.submenu')) {
      cancelActiveMenu();
    }
  }, { passive: true, capture: true });
}

function viewer(baseUrl, nodes) {
  'use strict';

  // Elements
  const search = document.getElementById('search');
  const graph0 = document.getElementById('graph0');
  const svg = (graph0 == null ? null : graph0.parentElement);
  const toptable = document.getElementById('toptable');

  let regexpActive = false;
  let selected = new Map();
  let origFill = new Map();
  let searchAlarm = null;
  let buttonsEnabled = true;

  function handleDetails(e) {
    e.preventDefault();
    const detailsText = document.getElementById('detailsbox');
    if (detailsText != null) {
      if (detailsText.style.display === 'block') {
        detailsText.style.display = 'none';
      } else {
        detailsText.style.display = 'block';
      }
    }
  }

  function handleKey(e) {
    if (e.keyCode != 13) return;
    window.location.href =
        updateUrl(new URL(window.location.href), 'f');
    e.preventDefault();
  }

  function handleSearch() {
    // Delay expensive processing so a flurry of key strokes is handled once.
    if (searchAlarm != null) {
      clearTimeout(searchAlarm);
    }
    searchAlarm = setTimeout(selectMatching, 300);

    regexpActive = true;
    updateButtons();
  }

  function selectMatching() {
    searchAlarm = null;
    let re = null;
    if (search.value != '') {
      try {
        re = new RegExp(search.value);
      } catch (e) {
        // TODO: Display error state in search box
        return;
      }
    }

    function match(text) {
      return re != null && re.test(text);
    }

    // drop currently selected items that do not match re.
    selected.forEach(function(v, n) {
      if (!match(nodes[n])) {
        unselect(n, document.getElementById('node' + n));
      }
    })

    // add matching items that are not currently selected.
    for (let n = 0; n < nodes.length; n++) {
      if (!selected.has(n) && match(nodes[n])) {
        select(n, document.getElementById('node' + n));
      }
    }

    updateButtons();
  }

  function toggleSvgSelect(elem) {
    // Walk up to immediate child of graph0
    while (elem != null && elem.parentElement != graph0) {
      elem = elem.parentElement;
    }
    if (!elem) return;

    // Disable regexp mode.
    regexpActive = false;

    const n = nodeId(elem);
    if (n < 0) return;
    if (selected.has(n)) {
      unselect(n, elem);
    } else {
      select(n, elem);
    }
    updateButtons();
  }

  function unselect(n, elem) {
    if (elem == null) return;
    selected.delete(n);
    setBackground(elem, false);
  }

  function select(n, elem) {
    if (elem == null) return;
    selected.set(n, true);
    setBackground(elem, true);
  }

  function nodeId(elem) {
    const id = elem.id;
    if (!id) return -1;
    if (!id.startsWith('node')) return -1;
    const n = parseInt(id.slice(4), 10);
    if (isNaN(n)) return -1;
    if (n < 0 || n >= nodes.length) return -1;
    return n;
  }

  function setBackground(elem, set) {
    // Handle table row highlighting.
    if (elem.nodeName == 'TR') {
      elem.classList.toggle('hilite', set);
      return;
    }

    // Handle svg element highlighting.
    const p = findPolygon(elem);
    if (p != null) {
      if (set) {
        origFill.set(p, p.style.fill);
        p.style.fill = '#ccccff';
      } else if (origFill.has(p)) {
        p.style.fill = origFill.get(p);
      }
    }
  }

  function findPolygon(elem) {
    if (elem.localName == 'polygon') return elem;
    for (const c of elem.children) {
      const p = findPolygon(c);
      if (p != null) return p;
    }
    return null;
  }

  // convert a string to a regexp that matches that string.
  function quotemeta(str) {
    return str.replace(/([\\\.?+*\[\](){}|^$])/g, '\\$1');
  }

  function setSampleIndexLink(id) {
    const elem = document.getElementById(id);
    if (elem != null) {
      setHrefParams(elem, function (params) {
        params.set("si", id);
      });
    }
  }

  // Update id's href to reflect current selection whenever it is
  // liable to be followed.
  function makeSearchLinkDynamic(id) {
    const elem = document.getElementById(id);
    if (elem == null) return;

    // Most links copy current selection into the 'f' parameter,
    // but Refine menu links are different.
    let param = 'f';
    if (id == 'ignore') param = 'i';
    if (id == 'hide') param = 'h';
    if (id == 'show') param = 's';
    if (id == 'show-from') param = 'sf';

    // We update on mouseenter so middle-click/right-click work properly.
    elem.addEventListener('mouseenter', updater);
    elem.addEventListener('touchstart', updater);

    function updater() {
      // The selection can be in one of two modes: regexp-based or
      // list-based.  Construct regular expression depending on mode.
      let re = regexpActive
        ? search.value
        : Array.from(selected.keys()).map(key => quotemeta(nodes[key])).join('|');

      setHrefParams(elem, function (params) {
        if (re != '') {
          // For focus/show/show-from, forget old parameter. For others, add to re.
          if (param != 'f' && param != 's' && param != 'sf' && params.has(param)) {
            const old = params.get(param);
            if (old != '') {
              re += '|' + old;
            }
          }
          params.set(param, re);
        } else {
          params.delete(param);
        }
      });
    }
  }

  function setHrefParams(elem, paramSetter) {
    let url = new URL(elem.href);
    url.hash = '';

    // Copy params from this page's URL.
    const params = url.searchParams;
    for (const p of new URLSearchParams(window.location.search)) {
      params.set(p[0], p[1]);
    }

    // Give the params to the setter to modify.
    paramSetter(params);

    elem.href = url.toString();
  }

  function handleTopClick(e) {
    // Walk back until we find TR and then get the Name column (index 5)
    let elem = e.target;
    while (elem != null && elem.nodeName != 'TR') {
      elem = elem.parentElement;
    }
    if (elem == null || elem.children.length < 6) return;

    e.preventDefault();
    const tr = elem;
    const td = elem.children[5];
    if (td.nodeName != 'TD') return;
    const name = td.innerText;
    const index = nodes.indexOf(name);
    if (index < 0) return;

    // Disable regexp mode.
    regexpActive = false;

    if (selected.has(index)) {
      unselect(index, elem);
    } else {
      select(index, elem);
    }
    updateButtons();
  }

  function updateButtons() {
    const enable = (search.value != '' || selected.size != 0);
    if (buttonsEnabled == enable) return;
    buttonsEnabled = enable;
    for (const id of ['focus', 'ignore', 'hide', 'show', 'show-from']) {
      const link = document.getElementById(id);
      if (link != null) {
        link.classList.toggle('disabled', !enable);
      }
    }
  }

  // Initialize button states
  updateButtons();

  // Setup event handlers
  initMenus();
  if (svg != null) {
    initPanAndZoom(svg, toggleSvgSelect);
  }
  if (toptable != null) {
    toptable.addEventListener('mousedown', handleTopClick);
    toptable.addEventListener('touchstart', handleTopClick);
  }

  const ids = ['topbtn', 'graphbtn', 'peek', 'list', 'disasm',
               'focus', 'ignore', 'hide', 'show', 'show-from'];
  ids.forEach(makeSearchLinkDynamic);

  const sampleIDs = [{{range .SampleTypes}}'{{.}}', {{end}}];
  sampleIDs.forEach(setSampleIndexLink);

  // Bind action to button with specified id.
  function addAction(id, action) {
    const btn = document.getElementById(id);
    if (btn != null) {
      btn.addEventListener('click', action);
      btn.addEventListener('touchstart', action);
    }
  }

  addAction('details', handleDetails);

  search.addEventListener('input', handleSearch);
  search.addEventListener('keydown', handleKey);

  // Give initial focus to main container so it can be scrolled using keys.
  const main = document.getElementById('bodycontainer');
  if (main) {
    main.focus();
  }
}
</script>
{{end}}

{{define "top" -}}
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{.Title}}</title>
  {{template "css" .}}
  <style type="text/css">
  </style>
</head>
<body>
  {{template "header" .}}
  <div id="top">
    <table id="toptable">
      <thead>
        <tr>
          <th id="flathdr1">Flat</th>
          <th id="flathdr2">Flat%</th>
          <th>Sum%</th>
          <th id="cumhdr1">Cum</th>
          <th id="cumhdr2">Cum%</th>
          <th id="namehdr">Name</th>
          <th>Inlined?</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  {{template "script" .}}
  <script>
    function makeTopTable(total, entries) {
      const rows = document.getElementById('rows');
      if (rows == null) return;

      // Store initial index in each entry so we have stable node ids for selection.
      for (let i = 0; i < entries.length; i++) {
        entries[i].Id = 'node' + i;
      }

      // Which column are we currently sorted by and in what order?
      let currentColumn = '';
      let descending = false;
      sortBy('Flat');

      function sortBy(column) {
        // Update sort criteria
        if (column == currentColumn) {
          descending = !descending; // Reverse order
        } else {
          currentColumn = column;
          descending = (column != 'Name');
        }

        // Sort according to current criteria.
        function cmp(a, b) {
          const av = a[currentColumn];
          const bv = b[currentColumn];
          if (av < bv) return -1;
          if (av > bv) return +1;
          return 0;
        }
        entries.sort(cmp);
        if (descending) entries.reverse();

        function addCell(tr, val) {
          const td = document.createElement('td');
          td.textContent = val;
          tr.appendChild(td);
        }

        function percent(v) {
          return (v * 100.0 / total).toFixed(2) + '%';
        }

        // Generate rows
        const fragment = document.createDocumentFragment();
        let sum = 0;
        for (const row of entries) {
          const tr = document.createElement('tr');
          tr.id = row.Id;
          sum += row.Flat;
          addCell(tr, row.FlatFormat);
          addCell(tr, percent(row.Flat));
          addCell(tr, percent(sum));
          addCell(tr, row.CumFormat);
          addCell(tr, percent(row.Cum));
          addCell(tr, row.Name);
          addCell(tr, row.InlineLabel);
          fragment.appendChild(tr);
        }

        rows.textContent = ''; // Remove old rows
        rows.appendChild(fragment);
      }

      // Make different column headers trigger sorting.
      function bindSort(id, column) {
        const hdr = document.getElementById(id);
        if (hdr == null) return;
        const fn = function() { sortBy(column) };
        hdr.addEventListener('click', fn);
        hdr.addEventListener('touch', fn);
      }
      bindSort('flathdr1', 'Flat');
      bindSort('flathdr2', 'Flat');
      bindSort('cumhdr1', 'Cum');
      bindSort('cumhdr2', 'Cum');
      bindSort('namehdr', 'Name');
    }

    viewer(new URL(window.location.href), {{.Nodes}});
    makeTopTable({{.Total}}, {{.Top}});
  </script>
</body>
</html>
{{end}}

{{define "sourcelisting" -}}
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{.Title}}</title>
  {{template "css" .}}
  {{template "weblistcss" .}}
  {{template "weblistjs" .}}
</head>
<body>
  {{template "header" .}}
  <div id="content" class="source">
    {{.HTMLBody}}
  </div>
  {{template "script" .}}
  <script>viewer(new URL(window.location.href), null);</script>
</body>
</html>
{{end}}

{{define "plaintext" -}}
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{.Title}}</title>
  {{template "css" .}}
</head>
<body>
  {{template "header" .}}
  <div id="content">
    <pre>
      {{.TextBody}}
    </pre>
  </div>
  {{template "script" .}}
  <script>viewer(new URL(window.location.href), null);</script>
</body>
</html>
{{end}}

{{define "flamegraph" -}}
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{.Title}}</title>
  {{template "css" .}}
  <style type="text/css">{{template "d3flamegraphcss" .}}</style>
  <style type="text/css">
    .flamegraph-content {
      width: 90%;
      min-width: 80%;
      margin-left: 5%;
    }
    .flamegraph-details {
      height: 1.2em;
      width: 90%;
      min-width: 90%;
      margin-left: 5%;
      padding: 15px 0 35px;
    }
  </style>
</head>
<body>
  {{template "header" .}}
  <div id="bodycontainer">
    <div id="flamegraphdetails" class="flamegraph-details"></div>
    <div class="flamegraph-content">
      <div id="chart"></div>
    </div>
  </div>
  {{template "script" .}}
  <script>viewer(new URL(window.location.href), {{.Nodes}});</script>
  <script>{{template "d3script" .}}</script>
  <script>{{template "d3flamegraphscript" .}}</script>
  <script>
    var data = {{.FlameGraph}};

    var width = document.getElementById('chart').clientWidth;

    var flameGraph = d3.flamegraph()
      .width(width)
      .cellHeight(18)
      .minFrameSize(1)
      .transitionDuration(750)
      .transitionEase(d3.easeCubic)
      .inverted(true)
      .title('')
      .tooltip(false)
      .details(document.getElementById('flamegraphdetails'));

    // <full name> (percentage, value)
    flameGraph.label((d) => d.data.f + ' (' + d.data.p + ', ' + d.data.l + ')');

    (function(flameGraph) {
      var oldColorMapper = flameGraph.color();
      function colorMapper(d) {
        // Hack to force default color mapper to use 'warm' color scheme by not passing libtype
        const { data, highlight } = d;
        return oldColorMapper({ data: { n: data.n }, highlight });
      }

      flameGraph.color(colorMapper);
    }(flameGraph));

    d3.select('#chart')
      .datum(data)
      .call(flameGraph);

    function clear() {
      flameGraph.clear();
    }

    function resetZoom() {
      flameGraph.resetZoom();
    }

    window.addEventListener('resize', function() {
      var width = document.getElementById('chart').clientWidth;
      var graphs = document.getElementsByClassName('d3-flame-graph');
      if (graphs.length > 0) {
        graphs[0].setAttribute('width', width);
      }
      flameGraph.width(width);
      flameGraph.resetZoom();
    }, true);

    var search = document.getElementById('search');
    var searchAlarm = null;

    function selectMatching() {
      searchAlarm = null;

      if (search.value != '') {
        flameGraph.search(search.value);
      } else {
        flameGraph.clear();
      }
    }

    function handleSearch() {
      // Delay expensive processing so a flurry of key strokes is handled once.
      if (searchAlarm != null) {
        clearTimeout(searchAlarm);
      }
      searchAlarm = setTimeout(selectMatching, 300);
    }

    search.addEventListener('input', handleSearch);
  </script>
</body>
</html>
{{end}}
`))
}
