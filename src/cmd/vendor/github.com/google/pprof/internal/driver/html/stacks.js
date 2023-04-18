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

  // Mapping from unit to a list of display scales/labels.
  // List should be ordered by increasing unit size.
  const UNITS = new Map([
    ['B', [
      ['B', 1],
      ['kB', Math.pow(2, 10)],
      ['MB', Math.pow(2, 20)],
      ['GB', Math.pow(2, 30)],
      ['TB', Math.pow(2, 40)],
      ['PB', Math.pow(2, 50)]]],
    ['s', [
      ['ns', 1e-9],
      ['µs', 1e-6],
      ['ms', 1e-3],
      ['s', 1],
      ['hrs', 60*60]]]]);

  // Fields
  let shownTotal = 0;       // Total value of all stacks
  let pivots = [];          // Indices of currently selected data.Sources entries.
  let matches = new Set();  // Indices of sources that match search
  let elems = new Map();    // Mapping from source index to display elements
  let displayList = [];     // List of boxes to display.
  let actionMenuOn = false; // Is action menu visible?
  let actionTarget = null;  // Box on which action menu is operating.

  // Setup to allow measuring text width.
  const textSizer = document.createElement('canvas');
  textSizer.id = 'textsizer';
  const textContext = textSizer.getContext('2d');

  // Get DOM elements.
  const chart = find('stack-chart');
  const search = find('search');
  const actions = find('action-menu');
  const actionTitle = find('action-title');
  const detailBox = find('current-details');

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
      for (let p of pivots) {
        r.set(p, true);
      }
      return r;
    }});

  render();

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
    let v = stacks.Sources[src].RE;
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
    url.searchParams.set('p', regexp);
    history.pushState('', '', url.toString()); // Makes back-button work
    matches = new Set();
    search.value = '';
    render();
  }

  function handleEnter(box, div) {
    if (actionMenuOn) return;
    const src = stacks.Sources[box.src];
    const d = details(box);
    div.title = d + ' ' + src.FullName + (src.Inlined ? "\n(inlined)" : "");
    detailBox.innerText = d;
    // Highlight all boxes that have the same source as box.
    toggleClass(box.src, 'hilite2', true);
  }

  function handleLeave(box) {
    if (actionMenuOn) return;
    detailBox.innerText = '';
    toggleClass(box.src, 'hilite2', false);
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
    const total = totalValue(places);
    const xscale = (width-2*PADDING) / total; // Converts from profile value to X pixels
    const x = PADDING;
    const y = 0;
    shownTotal = total;

    displayList.length = 0;
    renderStacks(0, xscale, x, y, places, +1);  // Callees
    renderStacks(0, xscale, x, y-ROW, places, -1);  // Callers (ROW left for separator)
    display(displayList);
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
      x += xscale*g.sum;
    }
  }

  function renderGroup(depth, xscale, x, y, g, direction) {
    // Skip if not wide enough.
    const width = xscale * g.sum;
    if (width < MIN_WIDTH) return;

    // Draw the box for g.src (except for selected element in upwards direction
    // since that duplicates the box we added in downwards direction).
    if (depth != 0 || direction > 0) {
      const box = {
        x:         x,
        y:         y,
        src:       g.src,
        sum:       g.sum,
        selfValue: g.self,
        width:     xscale*g.sum,
        selfWidth: (direction > 0) ? xscale*g.self : 0,
      };
      displayList.push(box);
      x += box.selfWidth;
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
        group = {name: name, src: src, sum: 0, self: 0, places: []};
        groupMap.set(src, group);
        groups.push(group);
      }
      group.sum += stack.Value;
      group.self += (place.Pos == stack.Sources.length-1) ? stack.Value : 0;
      group.places.push(place);
    }

    // Order by decreasing cost (makes it easier to spot heavy functions).
    // Though alphabetical ordering is a potential alternative that will make
    // profile comparisons easier.
    groups.sort(function(a, b) { return b.sum - a.sum; });

    return groups;
  }

  function display(list) {
    // Sort boxes so that text selection follows a predictable order.
    list.sort(function(a, b) {
      if (a.y != b.y) return a.y - b.y;
      return a.x - b.x;
    });

    // Adjust Y coordinates so that zero is at top.
    let adjust = (list.length > 0) ? list[0].y : 0;
    adjust -= ROW + 2*PADDING;  // Room for details

    const divs = [];
    for (const box of list) {
      box.y -= adjust;
      divs.push(drawBox(box));
    }
    divs.push(drawSep(-adjust));

    const h = (list.length > 0 ?  list[list.length-1].y : 0) + 4*ROW;
    chart.style.height = h+'px';
    chart.replaceChildren(...divs);
  }

  function drawBox(box) {
    const srcIndex = box.src;
    const src = stacks.Sources[srcIndex];

    // Background
    const w = box.width - 1; // Leave 1px gap
    const r = document.createElement('div');
    r.style.left = box.x + 'px';
    r.style.top = box.y + 'px';
    r.style.width = w + 'px';
    r.style.height = ROW + 'px';
    r.classList.add('boxbg');
    r.style.background = makeColor(src.Color);
    addElem(srcIndex, r);
    if (!src.Inlined) {
      r.classList.add('not-inlined');
    }

    // Box that shows time spent in self
    if (box.selfWidth >= MIN_WIDTH) {
      const s = document.createElement('div');
      s.style.width = Math.min(box.selfWidth, w)+'px';
      s.style.height = (ROW-1)+'px';
      s.classList.add('self');
      r.appendChild(s);
    }

    // Label
    if (box.width >= MIN_TEXT_WIDTH) {
      const t = document.createElement('div');
      t.classList.add('boxtext');
      fitText(t, box.width-2*TEXT_MARGIN, src.Display);
      r.appendChild(t);
    }

    r.addEventListener('click', () => { switchPivots(src.RE); });
    r.addEventListener('mouseenter', () => { handleEnter(box, r); });
    r.addEventListener('mouseleave', () => { handleLeave(box); });
    r.addEventListener('contextmenu', (e) => { showActionMenu(e, box); });
    return r;
  }

  function drawSep(y) {
    const m = document.createElement('div');
    m.innerText = percent(shownTotal, stacks.Total) +
	'\xa0\xa0\xa0\xa0' +  // Some non-breaking spaces
	valueString(shownTotal);
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

  // totalValue returns the combined sum of the stacks listed in places.
  function totalValue(places) {
    const seen = new Set();
    let result = 0;
    for (const place of places) {
      if (seen.has(place.Stack)) continue; // Do not double-count stacks
      seen.add(place.Stack);
      const stack = stacks.Stacks[place.Stack];
      result += stack.Value;
    }
    return result;
  }

  function details(box) {
    // E.g., 10% 7s
    // or    10% 7s (3s self
    let result = percent(box.sum, stacks.Total) + ' ' + valueString(box.sum);
    if (box.selfValue > 0) {
      result += ` (${valueString(box.selfValue)} self)`;
    }
    return result;
  }

  function percent(v, total) {
    return Number(((100.0 * v) / total).toFixed(1)) + '%';
  }

  // valueString returns a formatted string to display for value.
  function valueString(value) {
    let v = value * stacks.Scale;
    // Rescale to appropriate display unit.
    let unit = stacks.Unit;
    const list = UNITS.get(unit);
    if (list) {
      // Find first entry in list that is not too small.
      for (const [name, scale] of list) {
        if (v <= 100*scale) {
          v /= scale;
          unit = name;
          break;
        }
      }
    }
    return Number(v.toFixed(2)) + unit;
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
