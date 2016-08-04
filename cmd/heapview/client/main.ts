// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/**
 * A hamburger menu element.
 */
class HamburgerElement extends HTMLElement {
  connectedCallback() {
    this.innerHTML = '&#9776';  // Unicode character for hamburger menu.
  }
}
window.customElements.define('heap-hamburger', HamburgerElement);

/**
 * A heading for the page with a hamburger menu and a title.
 */
export class HeadingElement extends HTMLElement {
  connectedCallback() {
    this.style.display = 'block';
    this.style.backgroundColor = '#2196F3';
    this.style.webkitUserSelect = 'none';
    this.style.cursor = 'default';
    this.style.color = '#FFFFFF';
    this.style.padding = '10px';
    this.innerHTML = `
      <div style="margin:0px; font-size:2em"><heap-hamburger></heap-hamburger> Go Heap Viewer</div>
    `;
  }
}
window.customElements.define('heap-heading', HeadingElement);


/**
 * Reset body's margin and padding, and set font.
 */
function clearStyle() {
  document.head.innerHTML += `
  <style>
    * {font-family: Roboto,Helvetica}
    body {margin: 0px; padding:0px}
  </style>
  `;
}

export function main() {
  document.title = 'Go Heap Viewer';
  clearStyle();
  document.body.appendChild(new HeadingElement());
}
