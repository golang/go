// A D3.js plugin that produces flame graphs from hierarchical data.
// https://github.com/spiermar/d3-flame-graph
// Version 2.0.0-alpha4
// See LICENSE file for license details

package d3flamegraph

// JSSource returns the d3-flamegraph.js file
const JSSource = `
(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('d3')) :
	typeof define === 'function' && define.amd ? define(['exports', 'd3'], factory) :
	(factory((global.d3 = global.d3 || {}),global.d3));
}(this, (function (exports,d3) { 'use strict';

var d3__default = 'default' in d3 ? d3['default'] : d3;

var commonjsGlobal = typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};





function createCommonjsModule(fn, module) {
	return module = { exports: {} }, fn(module, module.exports), module.exports;
}

var d3Tip = createCommonjsModule(function (module) {
// d3.tip
// Copyright (c) 2013 Justin Palmer
//
// Tooltips for d3.js SVG visualizations

(function (root, factory) {
  if (typeof undefined === 'function' && undefined.amd) {
    // AMD. Register as an anonymous module with d3 as a dependency.
    undefined(['d3'], factory);
  } else if ('object' === 'object' && module.exports) {
    // CommonJS
    var d3$$1 = d3__default;
    module.exports = factory(d3$$1);
  } else {
    // Browser global.
    root.d3.tip = factory(root.d3);
  }
}(commonjsGlobal, function (d3$$1) {

  // Public - contructs a new tooltip
  //
  // Returns a tip
  return function() {
    var direction = d3_tip_direction,
        offset    = d3_tip_offset,
        html      = d3_tip_html,
        node      = initNode(),
        svg       = null,
        point     = null,
        target    = null;

    function tip(vis) {
      svg = getSVGNode(vis);
      point = svg.createSVGPoint();
      document.body.appendChild(node);
    }

    // Public - show the tooltip on the screen
    //
    // Returns a tip
    tip.show = function() {
      var args = Array.prototype.slice.call(arguments);
      if(args[args.length - 1] instanceof SVGElement) target = args.pop();

      var content = html.apply(this, args),
          poffset = offset.apply(this, args),
          dir     = direction.apply(this, args),
          nodel   = getNodeEl(),
          i       = directions.length,
          coords,
          scrollTop  = document.documentElement.scrollTop || document.body.scrollTop,
          scrollLeft = document.documentElement.scrollLeft || document.body.scrollLeft;

      nodel.html(content)
        .style('opacity', 1).style('pointer-events', 'all');

      while(i--) nodel.classed(directions[i], false);
      coords = direction_callbacks.get(dir).apply(this);
      nodel.classed(dir, true)
      	.style('top', (coords.top +  poffset[0]) + scrollTop + 'px')
      	.style('left', (coords.left + poffset[1]) + scrollLeft + 'px');

      return tip;
    };

    // Public - hide the tooltip
    //
    // Returns a tip
    tip.hide = function() {
      var nodel = getNodeEl();
      nodel.style('opacity', 0).style('pointer-events', 'none');
      return tip
    };

    // Public: Proxy attr calls to the d3 tip container.  Sets or gets attribute value.
    //
    // n - name of the attribute
    // v - value of the attribute
    //
    // Returns tip or attribute value
    tip.attr = function(n, v) {
      if (arguments.length < 2 && typeof n === 'string') {
        return getNodeEl().attr(n)
      } else {
        var args =  Array.prototype.slice.call(arguments);
        d3$$1.selection.prototype.attr.apply(getNodeEl(), args);
      }

      return tip
    };

    // Public: Proxy style calls to the d3 tip container.  Sets or gets a style value.
    //
    // n - name of the property
    // v - value of the property
    //
    // Returns tip or style property value
    tip.style = function(n, v) {
      if (arguments.length < 2 && typeof n === 'string') {
        return getNodeEl().style(n)
      } else {
        var args = Array.prototype.slice.call(arguments);
        d3$$1.selection.prototype.style.apply(getNodeEl(), args);
      }

      return tip
    };

    // Public: Set or get the direction of the tooltip
    //
    // v - One of n(north), s(south), e(east), or w(west), nw(northwest),
    //     sw(southwest), ne(northeast) or se(southeast)
    //
    // Returns tip or direction
    tip.direction = function(v) {
      if (!arguments.length) return direction
      direction = v == null ? v : functor(v);

      return tip
    };

    // Public: Sets or gets the offset of the tip
    //
    // v - Array of [x, y] offset
    //
    // Returns offset or
    tip.offset = function(v) {
      if (!arguments.length) return offset
      offset = v == null ? v : functor(v);

      return tip
    };

    // Public: sets or gets the html value of the tooltip
    //
    // v - String value of the tip
    //
    // Returns html value or tip
    tip.html = function(v) {
      if (!arguments.length) return html
      html = v == null ? v : functor(v);

      return tip
    };

    // Public: destroys the tooltip and removes it from the DOM
    //
    // Returns a tip
    tip.destroy = function() {
      if(node) {
        getNodeEl().remove();
        node = null;
      }
      return tip;
    };

    function d3_tip_direction() { return 'n' }
    function d3_tip_offset() { return [0, 0] }
    function d3_tip_html() { return ' ' }

    var direction_callbacks = d3$$1.map({
      n:  direction_n,
      s:  direction_s,
      e:  direction_e,
      w:  direction_w,
      nw: direction_nw,
      ne: direction_ne,
      sw: direction_sw,
      se: direction_se
    }),

    directions = direction_callbacks.keys();

    function direction_n() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.n.y - node.offsetHeight,
        left: bbox.n.x - node.offsetWidth / 2
      }
    }

    function direction_s() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.s.y,
        left: bbox.s.x - node.offsetWidth / 2
      }
    }

    function direction_e() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.e.y - node.offsetHeight / 2,
        left: bbox.e.x
      }
    }

    function direction_w() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.w.y - node.offsetHeight / 2,
        left: bbox.w.x - node.offsetWidth
      }
    }

    function direction_nw() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.nw.y - node.offsetHeight,
        left: bbox.nw.x - node.offsetWidth
      }
    }

    function direction_ne() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.ne.y - node.offsetHeight,
        left: bbox.ne.x
      }
    }

    function direction_sw() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.sw.y,
        left: bbox.sw.x - node.offsetWidth
      }
    }

    function direction_se() {
      var bbox = getScreenBBox();
      return {
        top:  bbox.se.y,
        left: bbox.e.x
      }
    }

    function initNode() {
      var node = d3$$1.select(document.createElement('div'));
      node.style('position', 'absolute').style('top', 0).style('opacity', 0)
      	.style('pointer-events', 'none').style('box-sizing', 'border-box');

      return node.node()
    }

    function getSVGNode(el) {
      el = el.node();
      if(el.tagName.toLowerCase() === 'svg')
        return el

      return el.ownerSVGElement
    }

    function getNodeEl() {
      if(node === null) {
        node = initNode();
        // re-add node to DOM
        document.body.appendChild(node);
      }
      return d3$$1.select(node);
    }

    // Private - gets the screen coordinates of a shape
    //
    // Given a shape on the screen, will return an SVGPoint for the directions
    // n(north), s(south), e(east), w(west), ne(northeast), se(southeast), nw(northwest),
    // sw(southwest).
    //
    //    +-+-+
    //    |   |
    //    +   +
    //    |   |
    //    +-+-+
    //
    // Returns an Object {n, s, e, w, nw, sw, ne, se}
    function getScreenBBox() {
      var targetel   = target || d3$$1.event.target;

      while ('undefined' === typeof targetel.getScreenCTM && 'undefined' === targetel.parentNode) {
          targetel = targetel.parentNode;
      }

      var bbox       = {},
          matrix     = targetel.getScreenCTM(),
          tbbox      = targetel.getBBox(),
          width      = tbbox.width,
          height     = tbbox.height,
          x          = tbbox.x,
          y          = tbbox.y;

      point.x = x;
      point.y = y;
      bbox.nw = point.matrixTransform(matrix);
      point.x += width;
      bbox.ne = point.matrixTransform(matrix);
      point.y += height;
      bbox.se = point.matrixTransform(matrix);
      point.x -= width;
      bbox.sw = point.matrixTransform(matrix);
      point.y -= height / 2;
      bbox.w  = point.matrixTransform(matrix);
      point.x += width;
      bbox.e = point.matrixTransform(matrix);
      point.x -= width / 2;
      point.y -= height / 2;
      bbox.n = point.matrixTransform(matrix);
      point.y += height;
      bbox.s = point.matrixTransform(matrix);

      return bbox
    }
    
    // Private - replace D3JS 3.X d3.functor() function
    function functor(v) {
    	return typeof v === "function" ? v : function() {
        return v
    	}
    }

    return tip
  };

}));
});

var flamegraph = function () {
  var w = 960; // graph width
  var h = null; // graph height
  var c = 18; // cell height
  var selection = null; // selection
  var tooltip = true; // enable tooltip
  var title = ''; // graph title
  var transitionDuration = 750;
  var transitionEase = d3.easeCubic; // tooltip offset
  var sort = false;
  var inverted = false; // invert the graph direction
  var clickHandler = null;
  var minFrameSize = 0;
  var details = null;

  var tip = d3Tip()
    .direction('s')
    .offset([8, 0])
    .attr('class', 'd3-flame-graph-tip')
    .html(function (d) { return label(d) });

  var svg;

  function name (d) {
    return d.data.n || d.data.name
  }

  function libtype (d) {
    return d.data.l || d.data.libtype
  }

  function children (d) {
    return d.c || d.children
  }

  function value (d) {
    return d.v || d.value
  }

  var label = function (d) {
    return name(d) + ' (' + d3.format('.3f')(100 * (d.x1 - d.x0), 3) + '%, ' + value(d) + ' samples)'
  };

  function setDetails (t) {
    if (details) { details.innerHTML = t; }
  }

  var colorMapper = function (d) {
    return d.highlight ? '#E600E6' : colorHash(name(d), libtype(d))
  };

  function generateHash (name) {
    // Return a vector (0.0->1.0) that is a hash of the input string.
    // The hash is computed to favor early characters over later ones, so
    // that strings with similar starts have similar vectors. Only the first
    // 6 characters are considered.
    const MAX_CHAR = 6;

    var hash = 0;
    var maxHash = 0;
    var weight = 1;
    var mod = 10;

    if (name) {
      for (var i = 0; i < name.length; i++) {
        if (i > MAX_CHAR) { break }
        hash += weight * (name.charCodeAt(i) % mod);
        maxHash += weight * (mod - 1);
        weight *= 0.70;
      }
      if (maxHash > 0) { hash = hash / maxHash; }
    }
    return hash
  }

  function colorHash (name, libtype) {
    // Return a color for the given name and library type. The library type
    // selects the hue, and the name is hashed to a color in that hue.

    var r;
    var g;
    var b;

    // Select hue. Order is important.
    var hue;
    if (typeof libtype === 'undefined' || libtype === '') {
      // default when libtype is not in use
      hue = 'warm';
    } else {
      hue = 'red';
      if (name.match(/::/)) {
        hue = 'yellow';
      }
      if (libtype === 'kernel') {
        hue = 'orange';
      } else if (libtype === 'jit') {
        hue = 'green';
      } else if (libtype === 'inlined') {
        hue = 'aqua';
      }
    }

    // calculate hash
    var vector = 0;
    if (name) {
      var nameArr = name.split('` + "`" + `');
      if (nameArr.length > 1) {
        name = nameArr[nameArr.length - 1]; // drop module name if present
      }
      name = name.split('(')[0]; // drop extra info
      vector = generateHash(name);
    }

    // calculate color
    if (hue === 'red') {
      r = 200 + Math.round(55 * vector);
      g = 50 + Math.round(80 * vector);
      b = g;
    } else if (hue === 'orange') {
      r = 190 + Math.round(65 * vector);
      g = 90 + Math.round(65 * vector);
      b = 0;
    } else if (hue === 'yellow') {
      r = 175 + Math.round(55 * vector);
      g = r;
      b = 50 + Math.round(20 * vector);
    } else if (hue === 'green') {
      r = 50 + Math.round(60 * vector);
      g = 200 + Math.round(55 * vector);
      b = r;
    } else if (hue === 'aqua') {
      r = 50 + Math.round(60 * vector);
      g = 165 + Math.round(55 * vector);
      b = g;
    } else {
      // original warm palette
      r = 200 + Math.round(55 * vector);
      g = 0 + Math.round(230 * (1 - vector));
      b = 0 + Math.round(55 * (1 - vector));
    }

    return 'rgb(' + r + ',' + g + ',' + b + ')'
  }

  function hide (d) {
    d.data.hide = true;
    if (children(d)) {
      children(d).forEach(hide);
    }
  }

  function show (d) {
    d.data.fade = false;
    d.data.hide = false;
    if (children(d)) {
      children(d).forEach(show);
    }
  }

  function getSiblings (d) {
    var siblings = [];
    if (d.parent) {
      var me = d.parent.children.indexOf(d);
      siblings = d.parent.children.slice(0);
      siblings.splice(me, 1);
    }
    return siblings
  }

  function hideSiblings (d) {
    var siblings = getSiblings(d);
    siblings.forEach(function (s) {
      hide(s);
    });
    if (d.parent) {
      hideSiblings(d.parent);
    }
  }

  function fadeAncestors (d) {
    if (d.parent) {
      d.parent.data.fade = true;
      fadeAncestors(d.parent);
    }
  }

  // function getRoot (d) {
  //   if (d.parent) {
  //     return getRoot(d.parent)
  //   }
  //   return d
  // }

  function zoom (d) {
    tip.hide(d);
    hideSiblings(d);
    show(d);
    fadeAncestors(d);
    update();
    if (typeof clickHandler === 'function') {
      clickHandler(d);
    }
  }

  function searchTree (d, term) {
    var re = new RegExp(term);
    var searchResults = [];

    function searchInner (d) {
      var label = name(d);

      if (children(d)) {
        children(d).forEach(function (child) {
          searchInner(child);
        });
      }

      if (label.match(re)) {
        d.highlight = true;
        searchResults.push(d);
      } else {
        d.highlight = false;
      }
    }

    searchInner(d);
    return searchResults
  }

  function clear (d) {
    d.highlight = false;
    if (children(d)) {
      children(d).forEach(function (child) {
        clear(child);
      });
    }
  }

  function doSort (a, b) {
    if (typeof sort === 'function') {
      return sort(a, b)
    } else if (sort) {
      return d3.ascending(name(a), name(b))
    }
  }

  var p = d3.partition();

  function filterNodes (root) {
    var nodeList = root.descendants();
    if (minFrameSize > 0) {
      var kx = w / (root.x1 - root.x0);
      nodeList = nodeList.filter(function (el) {
        return ((el.x1 - el.x0) * kx) > minFrameSize
      });
    }
    return nodeList
  }

  function update () {
    selection.each(function (root) {
      var x = d3.scaleLinear().range([0, w]);
      var y = d3.scaleLinear().range([0, c]);

      if (sort) root.sort(doSort);
      root.sum(function (d) {
        if (d.fade || d.hide) {
          return 0
        }
        // The node's self value is its total value minus all children.
        var v = value(d);
        if (children(d)) {
          var c = children(d);
          for (var i = 0; i < c.length; i++) {
            v -= value(c[i]);
          }
        }
        return v
      });
      p(root);

      var kx = w / (root.x1 - root.x0);
      function width (d) { return (d.x1 - d.x0) * kx }

      var descendants = filterNodes(root);
      var g = d3.select(this).select('svg').selectAll('g').data(descendants, function (d) { return d.id });

      g.transition()
        .duration(transitionDuration)
        .ease(transitionEase)
        .attr('transform', function (d) { return 'translate(' + x(d.x0) + ',' + (inverted ? y(d.depth) : (h - y(d.depth) - c)) + ')' });

      g.select('rect')
        .attr('width', width);

      var node = g.enter()
        .append('svg:g')
        .attr('transform', function (d) { return 'translate(' + x(d.x0) + ',' + (inverted ? y(d.depth) : (h - y(d.depth) - c)) + ')' });

      node.append('svg:rect')
        .transition()
        .delay(transitionDuration / 2)
        .attr('width', width);

      if (!tooltip) { node.append('svg:title'); }

      node.append('foreignObject')
        .append('xhtml:div');

      // Now we have to re-select to see the new elements (why?).
      g = d3.select(this).select('svg').selectAll('g').data(descendants, function (d) { return d.id });

      g.attr('width', width)
        .attr('height', function (d) { return c })
        .attr('name', function (d) { return name(d) })
        .attr('class', function (d) { return d.data.fade ? 'frame fade' : 'frame' });

      g.select('rect')
        .attr('height', function (d) { return c })
        .attr('fill', function (d) { return colorMapper(d) });

      if (!tooltip) {
        g.select('title')
          .text(label);
      }

      g.select('foreignObject')
        .attr('width', width)
        .attr('height', function (d) { return c })
        .select('div')
        .attr('class', 'd3-flame-graph-label')
        .style('display', function (d) { return (width(d) < 35) ? 'none' : 'block' })
        .transition()
        .delay(transitionDuration)
        .text(name);

      g.on('click', zoom);

      g.exit()
        .remove();

      g.on('mouseover', function (d) {
        if (tooltip) tip.show(d, this);
        setDetails(label(d));
      }).on('mouseout', function (d) {
        if (tooltip) tip.hide(d);
        setDetails('');
      });
    });
  }

  function merge (data, samples) {
    samples.forEach(function (sample) {
      var node = data.find(function (element) {
        return (element.name === sample.name)
      });

      if (node) {
        if (node.original) {
          node.original += sample.value;
        } else {
          node.value += sample.value;
        }
        if (sample.children) {
          if (!node.children) {
            node.children = [];
          }
          merge(node.children, sample.children);
        }
      } else {
        data.push(sample);
      }
    });
  }

  function s4 () {
    return Math.floor((1 + Math.random()) * 0x10000)
      .toString(16)
      .substring(1)
  }

  function injectIds (node) {
    node.id = s4() + '-' + s4() + '-' + '-' + s4() + '-' + s4();
    var children = node.c || node.children || [];
    for (var i = 0; i < children.length; i++) {
      injectIds(children[i]);
    }
  }

  function chart (s) {
    var root = d3.hierarchy(
      s.datum(), function (d) { return children(d) }
    );
    injectIds(root);
    selection = s.datum(root);

    if (!arguments.length) return chart

    if (!h) {
      h = (root.height + 2) * c;
    }

    selection.each(function (data) {
      if (!svg) {
        svg = d3.select(this)
          .append('svg:svg')
          .attr('width', w)
          .attr('height', h)
          .attr('class', 'partition d3-flame-graph')
          .call(tip);

        svg.append('svg:text')
          .attr('class', 'title')
          .attr('text-anchor', 'middle')
          .attr('y', '25')
          .attr('x', w / 2)
          .attr('fill', '#808080')
          .text(title);
      }
    });

    // first draw
    update();
  }

  chart.height = function (_) {
    if (!arguments.length) { return h }
    h = _;
    return chart
  };

  chart.width = function (_) {
    if (!arguments.length) { return w }
    w = _;
    return chart
  };

  chart.cellHeight = function (_) {
    if (!arguments.length) { return c }
    c = _;
    return chart
  };

  chart.tooltip = function (_) {
    if (!arguments.length) { return tooltip }
    if (typeof _ === 'function') {
      tip = _;
    }
    tooltip = !!_;
    return chart
  };

  chart.title = function (_) {
    if (!arguments.length) { return title }
    title = _;
    return chart
  };

  chart.transitionDuration = function (_) {
    if (!arguments.length) { return transitionDuration }
    transitionDuration = _;
    return chart
  };

  chart.transitionEase = function (_) {
    if (!arguments.length) { return transitionEase }
    transitionEase = _;
    return chart
  };

  chart.sort = function (_) {
    if (!arguments.length) { return sort }
    sort = _;
    return chart
  };

  chart.inverted = function (_) {
    if (!arguments.length) { return inverted }
    inverted = _;
    return chart
  };

  chart.label = function (_) {
    if (!arguments.length) { return label }
    label = _;
    return chart
  };

  chart.search = function (term) {
    var searchResults = [];
    selection.each(function (data) {
      searchResults = searchTree(data, term);
      update();
    });
    return searchResults
  };

  chart.clear = function () {
    selection.each(function (data) {
      clear(data);
      update();
    });
  };

  chart.zoomTo = function (d) {
    zoom(d);
  };

  chart.resetZoom = function () {
    selection.each(function (data) {
      zoom(data); // zoom to root
    });
  };

  chart.onClick = function (_) {
    if (!arguments.length) {
      return clickHandler
    }
    clickHandler = _;
    return chart
  };

  chart.merge = function (samples) {
    var newRoot; // Need to re-create hierarchy after data changes.
    selection.each(function (root) {
      merge([root.data], [samples]);
      newRoot = d3.hierarchy(root.data, function (d) { return children(d) });
      injectIds(newRoot);
    });
    selection = selection.datum(newRoot);
    update();
  };

  chart.color = function (_) {
    if (!arguments.length) { return colorMapper }
    colorMapper = _;
    return chart
  };

  chart.minFrameSize = function (_) {
    if (!arguments.length) { return minFrameSize }
    minFrameSize = _;
    return chart
  };

  chart.details = function (_) {
    if (!arguments.length) { return details }
    details = _;
    return chart
  };

  return chart
};

exports.flamegraph = flamegraph;

Object.defineProperty(exports, '__esModule', { value: true });

})));
`

// CSSSource returns the d3-flamegraph.css file
const CSSSource = `
.d3-flame-graph rect {
  stroke: #EEEEEE;
  fill-opacity: .8;
}

.d3-flame-graph rect:hover {
  stroke: #474747;
  stroke-width: 0.5;
  cursor: pointer;
}

.d3-flame-graph-label {
  pointer-events: none;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  font-size: 12px;
  font-family: Verdana;
  margin-left: 4px;
  margin-right: 4px;
  line-height: 1.5;
  padding: 0 0 0;
  font-weight: 400;
  color: black;
  text-align: left;
}

.d3-flame-graph .fade {
  opacity: 0.6 !important;
}

.d3-flame-graph .title {
  font-size: 20px;
  font-family: Verdana;
}

.d3-flame-graph-tip {
  line-height: 1;
  font-family: Verdana;
  font-size: 12px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  border-radius: 2px;
  pointer-events: none;
}

/* Creates a small triangle extender for the tooltip */
.d3-flame-graph-tip:after {
  box-sizing: border-box;
  display: inline;
  font-size: 10px;
  width: 100%;
  line-height: 1;
  color: rgba(0, 0, 0, 0.8);
  position: absolute;
  pointer-events: none;
}

/* Northward tooltips */
.d3-flame-graph-tip.n:after {
  content: "\25BC";
  margin: -1px 0 0 0;
  top: 100%;
  left: 0;
  text-align: center;
}

/* Eastward tooltips */
.d3-flame-graph-tip.e:after {
  content: "\25C0";
  margin: -4px 0 0 0;
  top: 50%;
  left: -8px;
}

/* Southward tooltips */
.d3-flame-graph-tip.s:after {
  content: "\25B2";
  margin: 0 0 1px 0;
  top: -8px;
  left: 0;
  text-align: center;
}

/* Westward tooltips */
.d3-flame-graph-tip.w:after {
  content: "\25B6";
  margin: -4px 0 0 -1px;
  top: 50%;
  left: 100%;
}
`
