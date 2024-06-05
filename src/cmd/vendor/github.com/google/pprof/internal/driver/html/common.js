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

function sendURL(method, url, done) {
  fetch(url.toString(), {method: method})
      .then((response) => { done(response.ok); })
      .catch((error) => { done(false); });
}

// Initialize handlers for saving/loading configurations.
function initConfigManager() {
  'use strict';

  // Initialize various elements.
  function elem(id) {
    const result = document.getElementById(id);
    if (!result) console.warn('element ' + id + ' not found');
    return result;
  }
  const overlay = elem('dialog-overlay');
  const saveDialog = elem('save-dialog');
  const saveInput = elem('save-name');
  const saveError = elem('save-error');
  const delDialog = elem('delete-dialog');
  const delPrompt = elem('delete-prompt');
  const delError = elem('delete-error');

  let currentDialog = null;
  let currentDeleteTarget = null;

  function showDialog(dialog) {
    if (currentDialog != null) {
      overlay.style.display = 'none';
      currentDialog.style.display = 'none';
    }
    currentDialog = dialog;
    if (dialog != null) {
      overlay.style.display = 'block';
      dialog.style.display = 'block';
    }
  }

  function cancelDialog(e) {
    showDialog(null);
  }

  // Show dialog for saving the current config.
  function showSaveDialog(e) {
    saveError.innerText = '';
    showDialog(saveDialog);
    saveInput.focus();
  }

  // Commit save config.
  function commitSave(e) {
    const name = saveInput.value;
    const url = new URL(document.URL);
    // Set path relative to existing path.
    url.pathname = new URL('./saveconfig', document.URL).pathname;
    url.searchParams.set('config', name);
    saveError.innerText = '';
    sendURL('POST', url, (ok) => {
      if (!ok) {
        saveError.innerText = 'Save failed';
      } else {
        showDialog(null);
        location.reload();  // Reload to show updated config menu
      }
    });
  }

  function handleSaveInputKey(e) {
    if (e.key === 'Enter') commitSave(e);
  }

  function deleteConfig(e, elem) {
    e.preventDefault();
    const config = elem.dataset.config;
    delPrompt.innerText = 'Delete ' + config + '?';
    currentDeleteTarget = elem;
    showDialog(delDialog);
  }

  function commitDelete(e, elem) {
    if (!currentDeleteTarget) return;
    const config = currentDeleteTarget.dataset.config;
    const url = new URL('./deleteconfig', document.URL);
    url.searchParams.set('config', config);
    delError.innerText = '';
    sendURL('DELETE', url, (ok) => {
      if (!ok) {
        delError.innerText = 'Delete failed';
        return;
      }
      showDialog(null);
      // Remove menu entry for this config.
      if (currentDeleteTarget && currentDeleteTarget.parentElement) {
        currentDeleteTarget.parentElement.remove();
      }
    });
  }

  // Bind event on elem to fn.
  function bind(event, elem, fn) {
    if (elem == null) return;
    elem.addEventListener(event, fn);
    if (event == 'click') {
      // Also enable via touch.
      elem.addEventListener('touchstart', fn);
    }
  }

  bind('click', elem('save-config'), showSaveDialog);
  bind('click', elem('save-cancel'), cancelDialog);
  bind('click', elem('save-confirm'), commitSave);
  bind('keydown', saveInput, handleSaveInputKey);

  bind('click', elem('delete-cancel'), cancelDialog);
  bind('click', elem('delete-confirm'), commitDelete);

  // Activate deletion button for all config entries in menu.
  for (const del of Array.from(document.getElementsByClassName('menu-delete-btn'))) {
    bind('click', del, (e) => {
      deleteConfig(e, del);
    });
  }
}

// options if present can contain:
//   hiliter: function(Number, Boolean): Boolean
//     Overridable mechanism for highlighting/unhighlighting specified node.
//   current: function() Map[Number,Boolean]
//     Overridable mechanism for fetching set of currently selected nodes.
function viewer(baseUrl, nodes, options) {
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

  // Return current selection.
  function getSelection() {
    if (selected.size > 0) {
      return selected;
    } else if (options && options.current) {
      return options.current();
    }
    return new Map();
  }

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
    setHrefParams(window.location, function (params) {
      params.set('f', search.value);
    });
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
        unselect(n);
      }
    })

    // add matching items that are not currently selected.
    if (nodes) {
      for (let n = 0; n < nodes.length; n++) {
        if (!selected.has(n) && match(nodes[n])) {
          select(n);
        }
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
      unselect(n);
    } else {
      select(n);
    }
    updateButtons();
  }

  function unselect(n) {
    if (setNodeHighlight(n, false)) selected.delete(n);
  }

  function select(n, elem) {
    if (setNodeHighlight(n, true)) selected.set(n, true);
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

  // Change highlighting of node (returns true if node was found).
  function setNodeHighlight(n, set) {
    if (options && options.hiliter) return options.hiliter(n, set);

    const elem = document.getElementById('node' + n);
    if (!elem) return false;

    // Handle table row highlighting.
    if (elem.nodeName == 'TR') {
      elem.classList.toggle('hilite', set);
      return true;
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

    return true;
  }

  function findPolygon(elem) {
    if (elem.localName == 'polygon') return elem;
    for (const c of elem.children) {
      const p = findPolygon(c);
      if (p != null) return p;
    }
    return null;
  }

  function setSampleIndexLink(si) {
    const elem = document.getElementById('sampletype-' + si);
    if (elem != null) {
      setHrefParams(elem, function (params) {
        params.set("si", si);
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
          : Array.from(getSelection().keys()).map(key => pprofQuoteMeta(nodes[key])).join('|');

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
    const enable = (search.value != '' || getSelection().size != 0);
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

  const ids = ['topbtn', 'graphbtn',
               'flamegraph',
               'peek', 'list',
               'disasm', 'focus', 'ignore', 'hide', 'show', 'show-from'];
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
  initConfigManager();

  search.addEventListener('input', handleSearch);
  search.addEventListener('keydown', handleKey);

  // Give initial focus to main container so it can be scrolled using keys.
  const main = document.getElementById('bodycontainer');
  if (main) {
    main.focus();
  }
}

// convert a string to a regexp that matches exactly that string.
function pprofQuoteMeta(str) {
  return '^' + str.replace(/([\\\.?+*\[\](){}|^$])/g, '\\$1') + '$';
}
