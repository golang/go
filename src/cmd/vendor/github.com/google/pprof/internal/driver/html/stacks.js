// stackViewer displays a flame-graph like view (extended to show callers).
//   stacks - report.StackSet
//   nodes  - List of names for each source in report.StackSet
function stackViewer(stacks, nodes) {
  'use strict';

  // Constants used in rendering.
  const ROW = 20;
  const PADDING = 2;
  const MIN_WIDTH = 4;
  const MIN_TEXT_WIDTH = 16;
  const TEXT_MARGIN = 2;
  const FONT_SIZE = 12;
  const MIN_FONT_SIZE = 8;

  // Fields
  let pivots = [];          // Indices of currently selected data.Sources entries.
  let matches = new Set();  // Indices of sources that match search
  let elems = new Map();    // Mapping from source index to display elements
  let displayList = [];     // List of boxes to display.
  let actionMenuOn = false; // Is action menu visible?
  let actionTarget = null;  // Box on which action menu is operating.
  let diff = false;         // Are we displaying a diff?
  let shown = 0;            // How many profile values are being displayed?

  for (const stack of stacks.Stacks) {
    if (stack.Value < 0) {
      diff = true;
      break;
    }
  }

  // Setup to allow measuring text width.
  const textSizer = document.createElement('canvas');
  textSizer.id = 'textsizer';
  const textContext = textSizer.getContext('2d');

  // Get DOM elements.
  const chart = find('stack-chart');
  const search = find('search');
  const actions = find('action-menu');
  const actionTitle = find('action-title');
  const leftDetailBox = find('current-details-left');
  const rightDetailBox = find('current-details-right');

  window.addEventListener('resize', render);
  window.addEventListener('popstate', render);
  search.addEventListener('keydown', handleSearchKey);

  // Withdraw action menu when clicking outside, or when item selected.
  document.addEventListener('mousedown', (e) => {
    if (!actions.contains(e.target)) {
      hideActionMenu();
    }
  });
  actions.addEventListener('click', hideActionMenu);

  // Initialize menus and other general UI elements.
  viewer(new URL(window.location.href), nodes, {
    hiliter: (n, on) => { return hilite(n, on); },
    current: () => {
      let r = new Map();
      if (pivots.length == 1 && pivots[0] == 0) {
        // Not pivoting
      } else {
        for (let p of pivots) {
          r.set(p, true);
        }
      }
      return r;
    }});

  render();
  clearDetails();

  // Helper functions follow:

  // hilite changes the highlighting of elements corresponding to specified src.
  function hilite(src, on) {
    if (on) {
      matches.add(src);
    } else {
      matches.delete(src);
    }
    toggleClass(src, 'hilite', on);
    return true;
  }

  // Display action menu (triggered by right-click on a frame)
  function showActionMenu(e, box) {
    if (box.src == 0) return; // No action menu for root
    e.preventDefault(); // Disable browser context menu
    const src = stacks.Sources[box.src];
    actionTitle.innerText = src.Display[src.Display.length-1];
    const menu = actions;
    menu.style.display = 'block';
    // Compute position so menu stays visible and near the mouse.
    const x = Math.min(e.clientX - 10, document.body.clientWidth - menu.clientWidth);
    const y = Math.min(e.clientY - 10, document.body.clientHeight - menu.clientHeight);
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
    // Set menu links to operate on clicked box.
    setHrefParam('action-source', 'f', box.src);
    setHrefParam('action-source-tab', 'f', box.src);
    setHrefParam('action-focus', 'f', box.src);
    setHrefParam('action-ignore', 'i', box.src);
    setHrefParam('action-hide', 'h', box.src);
    setHrefParam('action-showfrom', 'sf', box.src);
    toggleClass(box.src, 'hilite2', true);
    actionTarget = box;
    actionMenuOn = true;
  }

  function hideActionMenu() {
    actions.style.display = 'none';
    actionMenuOn = false;
    if (actionTarget != null) {
      toggleClass(actionTarget.src, 'hilite2', false);
    }
  }

  // setHrefParam updates the specified parameter in the  href of an <a>
  // element to make it operate on the specified src.
  function setHrefParam(id, param, src) {
    const elem = document.getElementById(id);
    if (!elem) return;

    let url = new URL(elem.href);
    url.hash = '';

    // Copy params from this page's URL.
    const params = url.searchParams;
    for (const p of new URLSearchParams(window.location.search)) {
      params.set(p[0], p[1]);
    }

    // Update params to include src.
    // When `pprof` is invoked with `-lines`, FullName will be suffixed with `:<line>`,
    // which we need to remove.
    let v = pprofQuoteMeta(stacks.Sources[src].FullName.replace(/:[0-9]+$/, ''));
    if (param != 'f' && param != 'sf') { // old f,sf values are overwritten
      // Add new source to current parameter value.
      const old = params.get(param);
      if (old && old != '') {
        v += '|' + old;
      }
    }
    params.set(param, v);

    elem.href = url.toString();
  }

  // Capture Enter key in the search box to make it pivot instead of focus.
  function handleSearchKey(e) {
    if (e.key != 'Enter') return;
    e.stopImmediatePropagation();  // Disable normal enter key handling
    const val = search.value;
    try {
      new RegExp(search.value);
    } catch (error) {
      return;  // TODO: Display error state in search box
    }
    switchPivots(val);
  }

  function switchPivots(regexp) {
    // Switch URL without hitting the server.
    const url = new URL(document.URL);
    if (regexp === '' || regexp === '^$') {
      url.searchParams.delete('p');  // Not pivoting
    } else {
      url.searchParams.set('p', regexp);
    }
    history.pushState('', '', url.toString()); // Makes back-button work
    matches = new Set();
    search.value = '';
    render();
  }

  function handleEnter(box, div) {
    if (actionMenuOn) return;
    const src = stacks.Sources[box.src];
    div.title = details(box) + ' │ ' + src.FullName + (src.Inlined ? "\n(inlined)" : "");
    leftDetailBox.innerText = src.FullName + (src.Inlined ? " (inlined)" : "");
    let timing = summary(box.sumpos, box.sumneg);
    if (box.self != 0) {
      timing = "self " + unitText(box.self) + " │ " + timing;
    }
    rightDetailBox.innerText = timing;
    // Highlight all boxes that have the same source as box.
    toggleClass(box.src, 'hilite2', true);
  }

  function handleLeave(box) {
    if (actionMenuOn) return;
    clearDetails();
    toggleClass(box.src, 'hilite2', false);
  }

  function clearDetails() {
    leftDetailBox.innerText = '';
    rightDetailBox.innerText = percentText(shown);
  }

  // Return list of sources that match the regexp given by the 'p' URL parameter.
  function urlPivots() {
    const pivots = [];
    const params = (new URL(document.URL)).searchParams;
    const val = params.get('p');
    if (val !== null && val != '') {
      try {
        const re = new RegExp(val);
        for (let i = 0; i < stacks.Sources.length; i++) {
          const src = stacks.Sources[i];
          if (re.test(src.UniqueName) || re.test(src.FileName)) {
            pivots.push(i);
          }
        }
      } catch (error) {}
    }
    if (pivots.length == 0) {
      pivots.push(0);
    }
    return pivots;
  }

  // render re-generates the stack display.
  function render() {
    pivots = urlPivots();

    // Get places where pivots occur.
    let places = [];
    for (let pivot of pivots) {
      const src = stacks.Sources[pivot];
      for (let p of src.Places) {
        places.push(p);
      }
    }

    const width = chart.clientWidth;
    elems.clear();
    actionTarget = null;
    const [pos, neg] = totalValue(places);
    const total = pos + neg;
    const xscale = (width-2*PADDING) / total; // Converts from profile value to X pixels
    const x = PADDING;
    const y = 0;

    // Show summary for pivots if we are actually pivoting.
    const showPivotSummary = !(pivots.length == 1 && pivots[0] == 0);

    shown = pos + neg;
    displayList.length = 0;
    renderStacks(0, xscale, x, y, places, +1);  // Callees
    renderStacks(0, xscale, x, y-ROW, places, -1);  // Callers (ROW left for separator)
    display(xscale, pos, neg, displayList, showPivotSummary);
  }

  // renderStacks creates boxes with top-left at x,y with children drawn as
  // nested stacks (below or above based on the sign of direction).
  // Returns the largest y coordinate filled.
  function renderStacks(depth, xscale, x, y, places, direction) {
    // Example: suppose we are drawing the following stacks:
    //   a->b->c
    //   a->b->d
    //   a->e->f
    // After rendering a, we will call renderStacks, with places pointing to
    // the preceding stacks.
    //
    // We first group all places with the same leading entry. In this example
    // we get [b->c, b->d] and [e->f]. We render the two groups side-by-side.
    const groups = partitionPlaces(places);
    for (const g of groups) {
      renderGroup(depth, xscale, x, y, g, direction);
      x += groupWidth(xscale, g);
    }
  }

  // Some of the types used below:
  //
  // // Group represents a displayed (sub)tree.
  // interface Group {
  //   name: string;     // Full name of source
  //   src: number;      // Index in stacks.Sources
  //   self: number;     // Contribution as leaf (may be < 0 for diffs)
  //   sumpos: number;   // Sum of |self| of positive nodes in tree (>= 0)
  //   sumneg: number;   // Sum of |self| of negative nodes in tree (>= 0)
  //   places: Place[];  // Stack slots that contributed to this group
  // }
  //
  // // Box is a rendered item.
  // interface Box {
  //   x: number;          // X coordinate of top-left
  //   y: number;          // Y coordinate of top-left
  //   width: number;      // Width of box to display
  //   src: number;        // Index in stacks.Sources
  //   sumpos: number;     // From corresponding Group
  //   sumneg: number;     // From corresponding Group
  //   self: number;       // From corresponding Group
  // };

  function groupWidth(xscale, g) {
    return xscale * (g.sumpos + g.sumneg);
  }

  function renderGroup(depth, xscale, x, y, g, direction) {
    // Skip if not wide enough.
    const width = groupWidth(xscale, g);
    if (width < MIN_WIDTH) return;

    // Draw the box for g.src (except for selected element in upwards direction
    // since that duplicates the box we added in downwards direction).
    if (depth != 0 || direction > 0) {
      const box = {
        x:      x,
        y:      y,
        width:  width,
        src:    g.src,
        sumpos: g.sumpos,
        sumneg: g.sumneg,
        self:   g.self,
      };
      displayList.push(box);
      if (direction > 0) {
        // Leave gap on left hand side to indicate self contribution.
        x += xscale*Math.abs(g.self);
      }
    }
    y += direction * ROW;

    // Find child or parent stacks.
    const next = [];
    for (const place of g.places) {
      const stack = stacks.Stacks[place.Stack];
      const nextSlot = place.Pos + direction;
      if (nextSlot >= 0 && nextSlot < stack.Sources.length) {
        next.push({Stack: place.Stack, Pos: nextSlot});
      }
    }
    renderStacks(depth+1, xscale, x, y, next, direction);
  }

  // partitionPlaces partitions a set of places into groups where each group
  // contains places with the same source. If a stack occurs multiple times
  // in places, only the outer-most occurrence is kept.
  function partitionPlaces(places) {
    // Find outer-most slot per stack (used later to elide duplicate stacks).
    const stackMap = new Map();  // Map from stack index to outer-most slot#
    for (const place of places) {
      const prevSlot = stackMap.get(place.Stack);
      if (prevSlot && prevSlot <= place.Pos) {
        // We already have a higher slot in this stack.
      } else {
        stackMap.set(place.Stack, place.Pos);
      }
    }

    // Now partition the stacks.
    const groups = [];           // Array of Group {name, src, sum, self, places}
    const groupMap = new Map();  // Map from Source to Group
    for (const place of places) {
      if (stackMap.get(place.Stack) != place.Pos) {
        continue;
      }

      const stack = stacks.Stacks[place.Stack];
      const src = stack.Sources[place.Pos];
      let group = groupMap.get(src);
      if (!group) {
        const name = stacks.Sources[src].FullName;
        group = {name: name, src: src, sumpos: 0, sumneg: 0, self: 0, places: []};
        groupMap.set(src, group);
        groups.push(group);
      }
      if (stack.Value < 0) {
        group.sumneg += -stack.Value;
      } else {
        group.sumpos += stack.Value;
      }
      group.self += (place.Pos == stack.Sources.length-1) ? stack.Value : 0;
      group.places.push(place);
    }

    // Order by decreasing cost (makes it easier to spot heavy functions).
    // Though alphabetical ordering is a potential alternative that will make
    // profile comparisons easier.
    groups.sort(function(a, b) {
      return (b.sumpos + b.sumneg) - (a.sumpos + a.sumneg);
    });

    return groups;
  }

  function display(xscale, posTotal, negTotal, list, showPivotSummary) {
    // Sort boxes so that text selection follows a predictable order.
    list.sort(function(a, b) {
      if (a.y != b.y) return a.y - b.y;
      return a.x - b.x;
    });

    // Adjust Y coordinates so that zero is at top.
    let adjust = (list.length > 0) ? list[0].y : 0;

    const divs = [];
    for (const box of list) {
      box.y -= adjust;
      divs.push(drawBox(xscale, box));
    }
    if (showPivotSummary) {
      divs.push(drawSep(-adjust, posTotal, negTotal));
    }

    const h = (list.length > 0 ?  list[list.length-1].y : 0) + 4*ROW;
    chart.style.height = h+'px';
    chart.replaceChildren(...divs);
  }

  function drawBox(xscale, box) {
    const srcIndex = box.src;
    const src = stacks.Sources[srcIndex];

    function makeRect(cl, x, y, w, h) {
      const r = document.createElement('div');
      r.style.left = x+'px';
      r.style.top = y+'px';
      r.style.width = w+'px';
      r.style.height = h+'px';
      r.classList.add(cl);
      return r;
    }

    // Background
    const w = box.width - 1; // Leave 1px gap
    const r = makeRect('boxbg', box.x, box.y, w, ROW);
    if (!diff) r.style.background = makeColor(src.Color);
    addElem(srcIndex, r);
    if (!src.Inlined) {
      r.classList.add('not-inlined');
    }

    // Positive/negative indicator for diff mode.
    if (diff) {
      const delta = box.sumpos - box.sumneg;
      const partWidth = xscale * Math.abs(delta);
      if (partWidth >= MIN_WIDTH) {
        r.appendChild(makeRect((delta < 0 ? 'negative' : 'positive'),
                               0, 0, partWidth, ROW-1));
      }
    }

    // Label
    if (box.width >= MIN_TEXT_WIDTH) {
      const t = document.createElement('div');
      t.classList.add('boxtext');
      fitText(t, box.width-2*TEXT_MARGIN, src.Display);
      r.appendChild(t);
    }

    onClick(r, () => { switchPivots(pprofQuoteMeta(src.UniqueName)); });
    r.addEventListener('mouseenter', () => { handleEnter(box, r); });
    r.addEventListener('mouseleave', () => { handleLeave(box); });
    r.addEventListener('contextmenu', (e) => { showActionMenu(e, box); });
    return r;
  }

  // Handle clicks, but only if the mouse did not move during the click.
  function onClick(target, handler) {
    // Disable click if mouse moves more than threshold pixels since mousedown.
    const threshold = 3;
    let [x, y] = [-1, -1];
    target.addEventListener('mousedown', (e) => {
      [x, y] = [e.clientX, e.clientY];
    });
    target.addEventListener('click', (e) => {
      if (Math.abs(e.clientX - x) <= threshold &&
          Math.abs(e.clientY - y) <= threshold) {
        handler();
      }
    });
  }

  function drawSep(y, posTotal, negTotal) {
    const m = document.createElement('div');
    m.innerText = summary(posTotal, negTotal);
    m.style.top = (y-ROW) + 'px';
    m.style.left = PADDING + 'px';
    m.style.width = (chart.clientWidth - PADDING*2) + 'px';
    m.classList.add('separator');
    return m;
  }

  // addElem registers an element that belongs to the specified src.
  function addElem(src, elem) {
    let list = elems.get(src);
    if (!list) {
      list = [];
      elems.set(src, list);
    }
    list.push(elem);
    elem.classList.toggle('hilite', matches.has(src));
  }

  // Adds or removes cl from classList of all elements for the specified source.
  function toggleClass(src, cl, value) {
    const list = elems.get(src);
    if (list) {
      for (const elem of list) {
        elem.classList.toggle(cl, value);
      }
    }
  }

  // fitText sets text and font-size clipped to the specified width w.
  function fitText(t, avail, textList) {
    // Find first entry in textList that fits.
    let width = avail;
    textContext.font = FONT_SIZE + 'pt Arial';
    for (let i = 0; i < textList.length; i++) {
      let text = textList[i];
      width = textContext.measureText(text).width;
      if (width <= avail) {
        t.innerText = text;
        return;
      }
    }

    // Try to fit by dropping font size.
    let text = textList[textList.length-1];
    const fs = Math.max(MIN_FONT_SIZE, FONT_SIZE * (avail / width));
    t.style.fontSize = fs + 'pt';
    t.innerText = text;
  }

  // totalValue returns the positive and negative sums of the Values of stacks
  // listed in places.
  function totalValue(places) {
    const seen = new Set();
    let pos = 0;
    let neg = 0;
    for (const place of places) {
      if (seen.has(place.Stack)) continue; // Do not double-count stacks
      seen.add(place.Stack);
      const stack = stacks.Stacks[place.Stack];
      if (stack.Value < 0) {
        neg += -stack.Value;
      } else {
        pos += stack.Value;
      }
    }
    return [pos, neg];
  }

  function summary(pos, neg) {
    // Examples:
    //    6s (10%)
    //    12s (20%) 🠆 18s (30%)
    return diff ? diffText(neg, pos) : percentText(pos);
  }

  function details(box) {
    // Examples:
    //    6s (10%)
    //    6s (10%) │ self 3s (5%)
    //    6s (10%) │ 12s (20%) 🠆 18s (30%)
    let result = percentText(box.sumpos - box.sumneg);
    if (box.self != 0) {
      result += " │ self " + unitText(box.self);
    }
    if (diff && box.sumpos > 0 && box.sumneg > 0) {
      result += " │ " + diffText(box.sumneg, box.sumpos);
    }
    return result;
  }

  // diffText returns text that displays from and to alongside their percentages.
  // E.g., 9s (45%) 🠆 10s (50%)
  function diffText(from, to) {
    return percentText(from) + " 🠆 " + percentText(to);
  }

  // percentText returns text that displays v in appropriate units alongside its
  // percentage.
  function percentText(v) {
    function percent(v, total) {
      return Number(((100.0 * v) / total).toFixed(1)) + '%';
    }
    return unitText(v) + " (" + percent(v, stacks.Total) + ")";
  }

  // unitText returns a formatted string to display for value.
  function unitText(value) {
    return pprofUnitText(value*stacks.Scale, stacks.Unit);
  }

  function find(name) {
    const elem = document.getElementById(name);
    if (!elem) {
      throw 'element not found: ' + name
    }
    return elem;
  }

  function makeColor(index) {
    // Rotate hue around a circle. Multiple by phi to spread things
    // out better. Use 50% saturation to make subdued colors, and
    // 80% lightness to have good contrast with black foreground text.
    const PHI = 1.618033988;
    const hue = (index+1) * PHI * 2 * Math.PI; // +1 to avoid 0
    const hsl = `hsl(${hue}rad 50% 80%)`;
    return hsl;
  }
}

// pprofUnitText returns a formatted string to display for value in the specified unit.
function pprofUnitText(value, unit) {
  const sign = (value < 0) ? "-" : "";
  let v = Math.abs(value);
  // Rescale to appropriate display unit.
  let list = null;
  for (const def of pprofUnitDefs) {
    if (def.DefaultUnit.CanonicalName == unit) {
      list = def.Units;
      v *= def.DefaultUnit.Factor;
      break;
    }
  }
  if (list) {
    // Stop just before entry that is too large.
    for (let i = 0; i < list.length; i++) {
      if (i == list.length-1 || list[i+1].Factor > v) {
        v /= list[i].Factor;
        unit = list[i].CanonicalName;
        break;
      }
    }
  }
  return sign + Number(v.toFixed(2)) + unit;
}
